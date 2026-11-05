use crate::{
    ir::NodeId,
    model::{MType, ModelOperation},
    tensor::{IRTrace, TensorIR, operation},
};

#[derive(Clone, Copy, Debug)]
pub struct Boundary(pub MType);

impl ModelOperation for Boundary {
    fn opname(&self) -> String {
        "Boundary".to_string()
    }

    fn inputs(&self) -> Vec<MType> {
        vec![self.0]
    }

    fn output(&self) -> MType {
        self.0
    }

    fn lower(&self, batch_size: usize, lower: &mut TensorIR, inputs: Vec<NodeId>) -> Result<NodeId, IRTrace> {
        let boundary = operation::Boundary(self.0.ttype(batch_size));
        lower.add_op(inputs, Ok::<_, IRTrace>(boundary)).map(|x| x[0])
    }
}
