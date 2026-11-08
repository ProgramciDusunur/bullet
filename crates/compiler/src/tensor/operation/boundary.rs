use crate::tensor::{IRTrace, OpType, TNode, TType, TValue};

#[derive(Debug)]
pub struct Boundary(pub TType);

impl OpType for Boundary {
    fn opname(&self) -> String {
        "boundary".to_string()
    }

    fn inputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn outputs(&self) -> Vec<TType> {
        vec![self.0]
    }

    fn evaluate(&self, inputs: Vec<&TValue>, mut outputs: Vec<&mut TValue>) -> bool {
        *outputs[0] = inputs[0].clone();
        true
    }

    fn backward<'a>(&self, _inputs: Vec<TNode<'a>>, output_grads: Vec<TNode<'a>>) -> Result<Vec<TNode<'a>>, IRTrace> {
        Ok(output_grads)
    }
}
