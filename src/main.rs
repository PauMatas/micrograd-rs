use std::ops::Add;
use std::ops::Mul;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, PartialEq, Clone)]
struct ValueNode {
    data: f64,
    _parents: Vec<Rc<ValueNode>>,
}

#[derive(Debug, PartialEq, Clone)]
struct Value(Rc<ValueNode>);

impl Value {
    fn new(data: f64) -> Self {
        Value(Rc::new(ValueNode {
            data,
            _parents: vec![],
        }))
    }
}

impl Deref for Value {
    type Target = ValueNode;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        Value(Rc::new(ValueNode { data: self.data + other.data, _parents: vec![self.0.clone(), other.0.clone()] }))
    }
}

impl Add<f64> for &Value {
    type Output = Value;

    fn add(self, other: f64) -> Value {
        let other_value = Value::new(other);
        Value(Rc::new(ValueNode { data: self.data + other_value.data, _parents: vec![self.0.clone(), other_value.0.clone()] }))
    }
}

impl Add<&Value> for f64 {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        other + self
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        Value(Rc::new(ValueNode { data: self.data * other.data, _parents: vec![self.0.clone(), other.0.clone()] }))
    }
}

impl Mul<f64> for &Value {
    type Output = Value;

    fn mul(self, other: f64) -> Value {
        let other_value = Value::new(other);
        Value(Rc::new(ValueNode { data: self.0.data * other_value.0.data, _parents: vec![self.0.clone(), other_value.0.clone()] }))
    }
}

impl Mul<&Value> for f64 {
    type Output = Value;

    fn mul(self, other: &Value) -> Value {
        other * self
    }
}

fn main() {
    let value = Value::new(1714.0);
    println!("{:?}", value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_data() {
        let x = Value::new(3.14);
        assert_eq!(x.data, 3.14);
    }

    #[test]
    fn test_value_sum() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let c = &a + &b;
        assert_eq!(c.data, 3.0);
    }

    #[test]
    fn test_value_sum_with_float() {
        let a = Value::new(1.0);
        let b = &a + 2.0;
        assert_eq!(b.data, 3.0);
        let c = 2.0 + &a;
        assert_eq!(c.data, 3.0);
    }

    #[test]
    fn test_value_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = &a * &b;
        assert_eq!(c.data, 6.0);
    }

    #[test]
    fn test_value_mul_with_float() {
        let a = Value::new(3.0);
        let b = &a * 2.0;
        assert_eq!(b.data, 6.0);
        let c = 2.0 * &a;
        assert_eq!(c.data, 6.0);
    }
}

