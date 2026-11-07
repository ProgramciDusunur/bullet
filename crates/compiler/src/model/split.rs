use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ir::{IRError, NodeId, Op},
    model::{
        Layout, MType, Model, ModelIR, ModelOp,
        operations::{
            Boundary, Broadcast, CReLU, Concat, Dim, Matmul, PointwiseBinary, ReLU, Reduce, SCReLU, SelectRows, Slice,
            SparseMatmul, SqrReLU,
        },
    },
    tensor::operation::{CABinary, Reduction},
};

const ONE_LAYER: &str =
    "Tensor parallelism cannot split the model: the feature transformer must feed exactly one dense layer!";

#[derive(PartialEq, Eq)]
pub enum WeightRole {
    Sharded { rows: usize, segments: Vec<usize> },
    SplitK,
    Head,
}

pub struct LocalModel {
    pub ir: ModelIR,
    pub loss: NodeId,
    pub partial_loss: Option<NodeId>,
}

pub struct ModelSplit {
    pub roles: BTreeMap<String, WeightRole>,
    pub ranks: Vec<LocalModel>,
}

pub fn split_model(ir: &ModelIR, loss: NodeId, ranks: usize) -> Result<ModelSplit, String> {
    assert!(ranks > 0);

    let ops = ir.ir.ordered_operations().map_err(|e| e.to_string())?;
    let mut analysis = Analysis::default();

    for op in &ops {
        analysis.visit(ir, op)?;
    }

    let contraction = analysis.check(ir)?;
    analysis.segments(ir, &ops, ranks)?;

    let mut roles = BTreeMap::new();
    for (id, (name, _)) in &ir.weights {
        let role = match analysis.segments.get(id) {
            Some(segments) => WeightRole::Sharded { rows: ir.node(*id).ty().rows, segments: segments.clone() },
            None if *id == contraction => WeightRole::SplitK,
            None => WeightRole::Head,
        };
        roles.insert(name.clone(), role);
    }

    let ranks = (0..ranks).map(|rank| analysis.local(ir, &ops, loss, ranks, rank)).collect::<Result<_, _>>();
    let ranks = ranks.map_err(|e| e.to_string())?;

    Ok(ModelSplit { roles, ranks })
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Origin {
    weight: NodeId,
    offset: isize,
}

impl Origin {
    fn shift(self, by: usize) -> Self {
        Self { weight: self.weight, offset: self.offset + by as isize }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Rep,
    Weight(Origin),
    Bias(Origin),
    Ft(Option<Origin>),
    Partial,
    Ps,
    Lm,
}

impl Class {
    fn is_rep(self) -> bool {
        matches!(self, Self::Rep | Self::Weight(_) | Self::Bias(_))
    }

    fn is_reg(self) -> bool {
        matches!(self, Self::Ps | Self::Lm)
    }

    fn max_reg(self, other: Self) -> Self {
        if self == Self::Lm || other == Self::Lm { Self::Lm } else { Self::Ps }
    }
}

#[derive(Default)]
struct Analysis {
    classes: BTreeMap<NodeId, Class>,
    leaves: BTreeMap<NodeId, BTreeSet<NodeId>>,
    links: BTreeMap<NodeId, (NodeId, isize)>,
    cuts: Vec<(NodeId, isize)>,
    split: BTreeSet<NodeId>,
    replicated: BTreeSet<NodeId>,
    contractions: Vec<NodeId>,
    end: Option<NodeId>,
    sparse: bool,
    segments: BTreeMap<NodeId, Vec<usize>>,
    weighted: BTreeSet<NodeId>,
}

impl Analysis {
    fn visit(&mut self, ir: &ModelIR, op: &Op<Model>) -> Result<(), String> {
        let out = op.outputs()[0];
        let ins = op.inputs();

        let class = if ins.is_empty() {
            if ir.weights.contains_key(&out) {
                self.leaves.insert(out, BTreeSet::from([out]));
                Class::Weight(Origin { weight: out, offset: 0 })
            } else {
                Class::Rep
            }
        } else {
            let mut classes = ins.iter().map(|id| self.classes[id]).collect::<Vec<_>>();

            if self.continues_chain(ir, op, &classes) {
                ins.iter().for_each(|&id| self.mark(id, false));
                Class::Partial
            } else {
                for (&id, class) in ins.iter().zip(&mut classes) {
                    if *class == Class::Partial {
                        self.end = Some(id);
                        *class = Class::Rep;
                    }
                }

                self.rule(ir, op, &classes)?
            }
        };

        if ir.weights.contains_key(&out) || ins.iter().any(|id| self.weighted.contains(id)) {
            self.weighted.insert(out);
        }

        self.classes.insert(out, class);

        Ok(())
    }

    fn continues_chain(&self, ir: &ModelIR, op: &Op<Model>, classes: &[Class]) -> bool {
        let data = op.data();
        let partials = classes.iter().filter(|&&class| class == Class::Partial).count();
        let single = op.inputs().iter().all(|&id| self.classes[&id] != Class::Partial || ir.node(id).children() == 1);

        let add = data.downcast::<PointwiseBinary>().is_some_and(|op| op.1 == CABinary::Add)
            && partials == 1
            && classes.iter().all(|class| *class == Class::Partial || class.is_rep());
        let select = data.downcast::<SelectRows>().is_some() && classes[0] == Class::Partial && classes[1].is_rep();

        single && (add || select)
    }

    fn rule(&mut self, ir: &ModelIR, op: &Op<Model>, classes: &[Class]) -> Result<Class, String> {
        let data = op.data();
        let ins = op.inputs();
        let out = op.outputs()[0];

        if let Some(&Slice(_, start, end, true)) = data.downcast::<Slice>() {
            match classes[0] {
                Class::Weight(origin) => {
                    self.cut(origin, start, end);
                    self.inherit(out, &[ins[0]]);
                    return Ok(Class::Weight(origin.shift(start)));
                }
                Class::Ft(origin) => {
                    origin.inspect(|&origin| self.cut(origin, start, end));
                    return Ok(Class::Ft(origin.map(|origin| origin.shift(start))));
                }
                _ => {}
            }
        }

        if let (Some(broadcast), Class::Weight(origin)) = (data.downcast::<Broadcast>(), classes[0]) {
            let class = match broadcast.1 {
                Dim::Cols => Some(Class::Weight(origin)),
                Dim::Batch => Some(Class::Bias(origin)),
                Dim::Rows => None,
            };

            if let Some(class) = class {
                self.inherit(out, &[ins[0]]);
                return Ok(class);
            }
        }

        if let Some(&PointwiseBinary(_, binary)) = data.downcast::<PointwiseBinary>() {
            let (add, mul) = (binary == CABinary::Add, binary == CABinary::Mul);

            match (classes[0], classes[1]) {
                (Class::Weight(a), Class::Weight(b)) if add && self.unify(a, b) => {
                    self.inherit(out, ins);
                    return Ok(Class::Weight(a));
                }
                (Class::Ft(a), Class::Ft(b)) if add || mul => return Ok(Class::Ft(self.unify_ft(a, b))),
                (Class::Ft(a), Class::Bias(b)) | (Class::Bias(b), Class::Ft(a)) if add => {
                    ins.iter().for_each(|&id| self.mark(id, true));
                    return Ok(Class::Ft(self.unify_ft(a, Some(b))));
                }
                (x, y)
                    if (add || mul)
                        && (x.is_reg() || y.is_reg())
                        && (x.is_rep() || y.is_rep())
                        && !(mul && ins.iter().any(|id| self.weighted.contains(id) && self.classes[id].is_rep())) =>
                {
                    ins.iter().for_each(|&id| self.mark(id, false));
                    return Ok(if mul { x.max_reg(y) } else { Class::Lm });
                }
                (x, y) if add && x.is_reg() && y.is_reg() => {
                    return Ok(if x == Class::Ps && y == Class::Ps { Class::Ps } else { Class::Lm });
                }
                _ => {}
            }
        }

        if data.downcast::<SparseMatmul>().is_some()
            && let (Class::Weight(origin), true) = (classes[0], classes[1].is_rep())
        {
            self.mark(ins[0], true);
            self.mark(ins[1], false);
            self.sparse = true;
            return Ok(Class::Ft(Some(origin)));
        }

        let is_activation = activation(data).is_some();

        match (classes, data.downcast::<Concat>(), data.downcast::<Matmul>(), data.downcast::<Reduce>()) {
            ([Class::Ft(origin)], ..) if is_activation => return Ok(Class::Ft(*origin)),
            ([Class::Ft(_), Class::Ft(_)], Some(_), ..) => return Ok(Class::Ft(None)),
            ([Class::Weight(_), Class::Ft(_)], _, Some(matmul), _)
                if ir.weights.contains_key(&ins[0]) && !matmul.lbatch && !matmul.ltrans && !matmul.rtrans =>
            {
                self.mark(ins[0], false);
                self.contractions.push(ins[0]);
                return Ok(Class::Partial);
            }
            ([Class::Ft(_)], .., Some(&Reduce(_, Dim::Rows, Reduction::Sum))) => return Ok(Class::Ps),
            ([class], .., Some(&Reduce(_, Dim::Batch, Reduction::Sum))) if class.is_reg() => return Ok(*class),
            _ => {}
        }

        if classes.iter().all(|class| class.is_rep()) {
            ins.iter().for_each(|&id| self.mark(id, false));
            Ok(Class::Rep)
        } else if classes.iter().any(|class| class.is_reg()) {
            Err("Tensor parallelism cannot split the model: a split regularizer must enter the loss linearly!".into())
        } else {
            Err(format!(
                "Tensor parallelism cannot split the model: {} is not supported on a split tensor!",
                data.0.opname()
            ))
        }
    }

    fn check(&self, ir: &ModelIR) -> Result<NodeId, String> {
        if !self.sparse {
            return Err("Tensor parallelism found no sparse feature transformer!".into());
        }

        let both = self.split.intersection(&self.replicated).map(|id| &ir.weights[id].0).min();
        let [contraction] = self.contractions[..] else { return Err(ONE_LAYER.into()) };

        if let Some(name) = both.or((ir.node(contraction).children() != 1).then(|| &ir.weights[&contraction].0)) {
            return Err(format!("Tensor parallelism cannot split '{name}': it is used both split and unsplit!"));
        }

        let mut sinks = ir.ir.nodes().filter(|node| node.children() == 0);
        if sinks.any(|node| matches!(self.classes[&node.id()], Class::Ft(_) | Class::Partial)) {
            return Err(ONE_LAYER.into());
        }

        self.end.ok_or(ONE_LAYER)?;
        Ok(contraction)
    }

    fn segments(&mut self, ir: &ModelIR, ops: &[Op<Model>], ranks: usize) -> Result<(), String> {
        let mut cuts = BTreeMap::<NodeId, BTreeSet<isize>>::new();

        for &weight in &self.split {
            let (root, offset) = self.find(weight);
            let rows = ir.node(weight).ty().rows as isize;
            cuts.entry(root).or_default().extend([offset, offset + rows]);
        }

        for &(weight, at) in self.cuts.iter().filter(|(weight, _)| self.split.contains(weight)) {
            let (root, offset) = self.find(weight);
            cuts.entry(root).or_default().insert(offset + at);
        }

        let range = |origin: Origin, len: usize| {
            let (root, offset) = self.find(origin.weight);
            let start = offset + origin.offset;
            let end = start + len as isize;
            let bounds = [start].into_iter().chain(cuts[&root].range(start + 1..end).copied()).chain([end]);
            bounds.collect::<Vec<_>>().windows(2).map(|w| (w[1] - w[0]) as usize).collect::<Vec<_>>()
        };

        let mut weights = self.split.iter().map(|id| (&ir.weights[id].0, *id)).collect::<Vec<_>>();
        weights.sort();

        for (name, weight) in weights {
            let rows = ir.node(weight).ty().rows;
            if let Some(len) = range(Origin { weight, offset: 0 }, rows).into_iter().find(|len| len % (4 * ranks) != 0)
            {
                let multiple = 4 * ranks;
                return Err(format!(
                    "Tensor parallelism cannot split '{name}': segment of length {len} is not a multiple of {multiple}!"
                ));
            }
        }

        let mut segments = BTreeMap::<NodeId, Vec<usize>>::new();

        for op in ops {
            let out = op.outputs()[0];
            let rows = ir.node(out).ty().rows;
            let split = |id: &NodeId| self.leaves.get(id).is_some_and(|leaves| leaves.is_subset(&self.split));
            let inputs = op.inputs().iter().filter_map(|id| segments.get(id)).cloned().collect::<Vec<_>>();

            let segs = match self.classes[&out] {
                Class::Weight(origin) | Class::Bias(origin) if split(&out) => range(origin, rows),
                Class::Ft(Some(origin)) => range(origin, rows),
                Class::Ft(None) => {
                    if op.data().downcast::<Concat>().is_some() {
                        inputs.concat()
                    } else if let Some(&Slice(_, start, end, _)) = op.data().downcast::<Slice>() {
                        let bounds = inputs[0].iter().scan(0, |at, len| {
                            *at += len;
                            Some(*at)
                        });
                        let bounds = [0].into_iter().chain(bounds).collect::<Vec<_>>();
                        let (Some(i), Some(j)) = (bounds.binary_search(&start).ok(), bounds.binary_search(&end).ok())
                        else {
                            return Err(format!(
                                "Tensor parallelism cannot split the model: \
                                slice {start}..{end} is not aligned to the split!"
                            ));
                        };
                        inputs[0][i..j].to_vec()
                    } else if inputs.windows(2).all(|w| w[0] == w[1]) {
                        inputs[0].clone()
                    } else {
                        let opname = op.data().0.opname();
                        return Err(format!(
                            "Tensor parallelism cannot split the model: {opname} is not supported on a split tensor!"
                        ));
                    }
                }
                _ => continue,
            };

            segments.insert(out, segs);
        }

        self.segments = segments;
        Ok(())
    }

    fn local(
        &self,
        ir: &ModelIR,
        ops: &[Op<Model>],
        loss: NodeId,
        ranks: usize,
        rank: usize,
    ) -> Result<LocalModel, IRError> {
        let mut local = ModelIR::default();
        let mut map = BTreeMap::new();
        let mut partials = BTreeMap::<NodeId, (NodeId, Option<NodeId>)>::new();
        let mut regs = BTreeMap::new();

        let rows =
            |id: &NodeId| self.segments.get(id).map_or(ir.node(*id).ty().rows, |s| s.iter().sum::<usize>() / ranks);
        let ty = |id: &NodeId| MType { rows: rows(id), ..ir.node(*id).ty() };

        for op in ops {
            let (out, data, ins) = (op.outputs()[0], op.data(), op.inputs());
            let mapped = ins.iter().filter_map(|id| map.get(id).copied()).collect::<Vec<_>>();

            if ins.is_empty() {
                let node = match (ir.weights.get(&out), ir.inputs.get(&out)) {
                    (Some((name, init)), _) => local.add_weight(name, rows(&out), ty(&out).cols, init.clone()),
                    (_, Some(name)) => {
                        let MType { batch, rows, cols, layout } = ty(&out);
                        local.add_input(name, batch, rows, cols, layout)
                    }
                    _ => local.add_op([], data.clone())?,
                };
                map.insert(out, node);
                continue;
            }

            if self.classes[&out] == Class::Partial {
                let (p, r) = if let Some(matmul) = data.downcast::<Matmul>() {
                    let blocks = blocks(&self.segments[&ins[1]], ranks, rank);
                    (contract(&mut local, ir.node(out).ty(), matmul, map[&ins[0]], map[&ins[1]], &blocks)?, None)
                } else if data.downcast::<SelectRows>().is_some() {
                    let ((p, r), indices) = (partials[&ins[0]], map[&ins[1]]);
                    let r = r.map(|r| local.add_op([r, indices], data.clone())).transpose()?;
                    (local.add_op([p, indices], data.clone())?, r)
                } else {
                    let (partial, other) =
                        if partials.contains_key(&ins[0]) { (ins[0], ins[1]) } else { (ins[1], ins[0]) };
                    let ((p, r), other) = (partials[&partial], map[&other]);
                    let r = match r {
                        Some(r) => local.add_op([r, other], data.clone())?,
                        None => other,
                    };
                    (p, Some(r))
                };

                partials.insert(out, (p, r));

                if Some(out) == self.end {
                    let out_ty = ir.node(out).ty();
                    let mut z = local.add_op([p], Boundary(out_ty))?;
                    if let Some(r) = r {
                        z = local.add_op([z, r], PointwiseBinary(out_ty, CABinary::Add))?;
                    }
                    map.insert(out, z);
                }

                continue;
            }

            let node = if self.segments.contains_key(&out) || ins.iter().any(|id| self.segments.contains_key(id)) {
                let op: ModelOp = if let Some(&Slice(_, start, end, true)) = data.downcast::<Slice>() {
                    let segs = &self.segments[&ins[0]];
                    let at = |x: usize| {
                        let mut acc = 0;
                        let before = segs.iter().take_while(|&&len| {
                            acc += len;
                            acc <= x
                        });
                        before.map(|len| len / ranks).sum()
                    };
                    Slice::new(ty(&ins[0]), at(start), at(end), true).into()
                } else if let Some(&Broadcast(_, dim, reps)) = data.downcast::<Broadcast>() {
                    Broadcast(ty(&ins[0]), dim, reps).into()
                } else if let Some(&PointwiseBinary(_, binary)) = data.downcast::<PointwiseBinary>() {
                    PointwiseBinary(ty(&out), binary).into()
                } else if let Some(matmul) = data.downcast::<SparseMatmul>() {
                    SparseMatmul { rows: rows(&out), ..*matmul }.into()
                } else if let Some(&Reduce(_, dim, reduction)) = data.downcast::<Reduce>() {
                    Reduce(ty(&ins[0]), dim, reduction).into()
                } else if data.downcast::<Concat>().is_some() {
                    let Layout::Dense(dtype) = ty(&out).layout else { unreachable!() };
                    Concat::new(dtype, rows(&ins[0]), rows(&ins[1]), ty(&out).batch).into()
                } else {
                    let Some(rebuild) = activation(data) else { unreachable!() };
                    rebuild(ty(&out))
                };

                local.add_op(&mapped, op)?
            } else {
                local.add_op(&mapped, data.clone())?
            };

            map.insert(out, node);

            match self.classes[&out] {
                Class::Ps => {
                    regs.insert(out, node);
                }
                Class::Lm => {
                    let parts = ins.iter().filter_map(|id| regs.get(id).copied()).collect::<Vec<_>>();
                    let is_add = data.downcast::<PointwiseBinary>().is_some_and(|op| op.1 == CABinary::Add);

                    let reg = if is_add && parts.len() == 1 {
                        parts[0]
                    } else if is_add {
                        local.add_op(&parts, data.clone())?
                    } else {
                        let inputs = ins.iter().map(|id| regs.get(id).copied().unwrap_or(map[id])).collect::<Vec<_>>();
                        local.add_op(&inputs, data.clone())?
                    };

                    regs.insert(out, reg);
                }
                _ => {}
            }
        }

        Ok(LocalModel { ir: local, loss: map[&loss], partial_loss: regs.get(&loss).copied() })
    }

    fn mark(&mut self, node: NodeId, split: bool) {
        if let Class::Weight(_) | Class::Bias(_) = self.classes[&node] {
            let set = if split { &mut self.split } else { &mut self.replicated };
            set.extend(self.leaves[&node].iter().copied());
        }
    }

    fn inherit(&mut self, out: NodeId, from: &[NodeId]) {
        let leaves = from.iter().flat_map(|id| self.leaves[id].iter().copied()).collect();
        self.leaves.insert(out, leaves);
    }

    fn cut(&mut self, origin: Origin, start: usize, end: usize) {
        self.cuts.push((origin.weight, origin.shift(start).offset));
        self.cuts.push((origin.weight, origin.shift(end).offset));
    }

    fn find(&self, weight: NodeId) -> (NodeId, isize) {
        let (mut node, mut offset) = (weight, 0);

        while let Some(&(parent, shift)) = self.links.get(&node) {
            (node, offset) = (parent, offset + shift);
        }

        (node, offset)
    }

    fn unify(&mut self, a: Origin, b: Origin) -> bool {
        let ((ra, da), (rb, db)) = (self.find(a.weight), self.find(b.weight));
        let (pa, pb) = (da + a.offset, db + b.offset);

        if ra == rb {
            return pa == pb;
        }

        self.links.insert(rb, (ra, pa - pb));
        true
    }

    fn unify_ft(&mut self, a: Option<Origin>, b: Option<Origin>) -> Option<Origin> {
        let (a, b) = (a?, b?);
        self.unify(a, b).then_some(a)
    }
}

fn blocks(segments: &[usize], ranks: usize, rank: usize) -> Vec<(usize, usize, usize)> {
    let mut blocks = Vec::<(usize, usize, usize)>::new();
    let (mut start, mut local) = (0, 0);

    for len in segments {
        let block = len / ranks;
        let (col, row) = (start + rank * block, local);

        match blocks.last_mut() {
            Some(last) if last.0 + last.2 == col && last.1 + last.2 == row => last.2 += block,
            _ => blocks.push((col, row, block)),
        }

        start += len;
        local += block;
    }

    blocks
}

fn contract(
    local: &mut ModelIR,
    out: MType,
    matmul: &Matmul,
    lhs: NodeId,
    rhs: NodeId,
    blocks: &[(usize, usize, usize)],
) -> Result<NodeId, IRError> {
    let (lhs_ty, rhs_ty) = (local.node(lhs).ty(), local.node(rhs).ty());
    let mut sum = None;

    for &(col, row, len) in blocks {
        let a = if len == lhs_ty.cols {
            lhs
        } else {
            local.add_op([lhs], Slice::new(lhs_ty, col, col + len, false))?
        };
        let x = if len == rhs_ty.rows {
            rhs
        } else {
            local.add_op([rhs], Slice::new(rhs_ty, row, row + len, true))?
        };
        let term = local.add_op([a, x], Matmul { n: len, ..*matmul })?;

        sum = Some(match sum {
            None => term,
            Some(sum) => local.add_op([sum, term], PointwiseBinary(out, CABinary::Add))?,
        });
    }

    Ok(sum.unwrap())
}

fn activation(op: &ModelOp) -> Option<fn(MType) -> ModelOp> {
    if op.downcast::<ReLU>().is_some() {
        Some(|ty| ReLU(ty).into())
    } else if op.downcast::<CReLU>().is_some() {
        Some(|ty| CReLU(ty).into())
    } else if op.downcast::<SCReLU>().is_some() {
        Some(|ty| SCReLU(ty).into())
    } else if op.downcast::<SqrReLU>().is_some() {
        Some(|ty| SqrReLU(ty).into())
    } else {
        None
    }
}
