use std::collections::BTreeSet;

use crate::{
    ir::NodeId,
    tensor::{IRTrace, TensorIR, operation::Boundary, transform::eliminate::EliminateUnusedOperations},
};

pub struct BoundarySplit {
    pub before: TensorIR,
    pub after: TensorIR,
    pub partial: NodeId,
    pub reduced: NodeId,
    pub cross: Vec<NodeId>,
}

pub fn split_at_boundary(ir: &TensorIR) -> Result<BoundarySplit, IRTrace> {
    let boundaries = ir.ir().operations().filter(|op| op.data().downcast::<Boundary>().is_some()).collect::<Vec<_>>();
    let [boundary] = boundaries[..] else {
        return Err(format!("Expected exactly one boundary, found {}!", boundaries.len()).into());
    };

    let (partial, reduced) = (boundary.inputs()[0], boundary.outputs()[0]);
    let ancestors = ir.get_dependent_ops_set(ir.get_parent_op(partial)?)?;

    let mut cross = BTreeSet::new();
    for op in ir.ir().operations().filter(|op| op.id() != boundary.id() && !ancestors.contains(&op.id())) {
        for &input in op.inputs() {
            if input == partial {
                return Err("Boundary input is read past the boundary!".into());
            }

            let parent = ir.get_op(ir.get_parent_op(input)?)?;
            if ancestors.contains(&parent.id()) && !parent.inputs().is_empty() {
                cross.insert(input);
            }
        }
    }

    let mut before = ir.clone();
    before.outputs.clear();
    before.register_output(partial);
    cross.iter().for_each(|&id| before.register_output(id));
    before.transform(EliminateUnusedOperations)?;

    let mut after = ir.clone();
    for &id in [reduced].iter().chain(&cross) {
        let input = after.add_input(after.get_node(id)?.ty());
        after.swap_outputs(input, id)?;
    }
    after.transform(EliminateUnusedOperations)?;

    Ok(BoundarySplit { before, after, partial, reduced, cross: cross.into_iter().collect() })
}
