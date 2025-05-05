use std::fmt::Debug;
use std::fmt::Display;
use std::ops::Add;
//use std::ops::Deref;
//use std::ops::Mul;
use std::rc::Rc;
use std::cell::RefCell;
use std::vec;

struct Value {
    data: f64,
    grad: f64,
    _parents: Vec<Rc<RefCell<Value>>>,
    _operation: &'static str,
    backward: Box<dyn FnMut()>,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

#[derive(Clone)]
struct ValueRef {
    inner: Rc<RefCell<Value>>,
}

impl Value {
    fn new(data: f64) -> ValueRef {
        ValueRef {
            inner: Rc::new(RefCell::new(Value {
                data,
                grad: 0.0,
                _operation: "",
                _parents: vec![],
                backward: Box::new(|| {}),
            }))
        }
    }
}

impl ValueRef {
    pub fn set_grad(&self, new_grad: f64) {
        let mut value = self.inner.borrow_mut();
        value.grad = new_grad;

        let parents = value._parents.clone();

        if value._operation == "+" {
            value.backward = Box::new(move || {
                for p in &parents {
                    p.borrow_mut().grad += new_grad
                }
            })
        } else if value._operation == "*" {
            todo!()
        } else {
            todo!()
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value {{ data: {}, grad: {}, op: '{}' }}",
            self.data, self.grad, self._operation
        )?;

        if !self._parents.is_empty() {
            write!(f, ", parents: [")?;
            for parent_rc in self._parents.iter() {
                write!(f, " {},", parent_rc.borrow().data)?;
            }
            write!(f, " ]")?;
        }
        write!(f, " }}")
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Value {{ data: {}, grad: {} }}",
            self.data, self.grad
        )
    }
}

impl Debug for ValueRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.inner.borrow())
    }
}

impl Display for ValueRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner.borrow())
    }
}

impl Add for &ValueRef {
    type Output = ValueRef;

    fn add(self, other: &ValueRef) -> ValueRef {
        let out = Rc::new(RefCell::new(Value {
            data: self.inner.borrow().data + other.inner.borrow().data,
            grad: 0.0,
            _operation: "+",
            _parents: vec![self.inner.clone(), other.inner.clone()],
            backward: Box::new(|| {}),
        }));

        let a = self.clone();
        let b = other.clone();
        let out_rc = out.clone();

        out.borrow_mut().backward = Box::new(move || {
            let g = out_rc.borrow().grad;
            a.inner.borrow_mut().grad += g;
            b.inner.borrow_mut().grad += g;
        });
        
        ValueRef { inner: out }
    }
}

//impl Add<f64> for &Value {
//    type Output = Value;
//
//    fn add(self, other: f64) -> Value {
//        let other_value = Value::new(other);
//        self + &other_value
//    }
//}
//
//impl Add<&Value> for f64 {
//    type Output = Value;
//
//    fn add(self, other: &Value) -> Value {
//        other + self
//    }
//}
//
//impl Mul for &Value {
//    type Output = Value;
//
//    fn mul(self, other: &Value) -> Value {
//        let out = Value(Rc::new(ValueNode {
//            data: self.data * other.data,
//            grad: 0.0,
//            _parents: vec![self.0.clone(), other.0.clone()],
//            _operation: "*",
//            backward: Box::new(|| {})
//        }));
//
//        let self_ = self.clone();
//        let other_ = other.clone();
//        let out_ = out.clone();
//
//        out.0.borrow_mut().backward = Box::new(move || {
//            let out_grad = out_.0.borrow().grad;
//            let self_data = self_.0.borrow().data;
//            let other_data = other_.0.borrow().data;
//
//            self_.0.borrow_mut().grad += other_data * out_grad;
//            other_.0.borrow_mut().grad += self_data * out_grad;
//        });
//
//        out
//    }
//}
//
//impl Mul<f64> for &Value {
//    type Output = Value;
//
//    fn mul(self, other: f64) -> Value {
//        let other_value = Value::new(other);
//        self * &other_value
//    }
//}
//
//impl Mul<&Value> for f64 {
//    type Output = Value;
//
//    fn mul(self, other: &Value) -> Value {
//        other * self
//    }
//}

fn main() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);

    let c_value = Rc::new(RefCell::new(Value {
        data: a.inner.borrow().data + b.inner.borrow().data,
        grad: 0.0,
        _operation: "+",
        _parents: vec![a.inner.clone(), b.inner.clone()],
        backward: Box::new(|| {}),
    }));

    let c = ValueRef { inner: c_value };
    c.set_grad(1.0);
    
    (c.inner.borrow_mut().backward)();

    println!("a: {}, b: {}", a, b);
    println!("{:?}", c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valueref_data() {
        let x: ValueRef = Value::new(3.14);
        assert_eq!(x.inner.borrow().data, 3.14);
    }

    #[test]
    fn test_valueref_sum() {
        let a: ValueRef = Value::new(1.0);
        let b: ValueRef = Value::new(2.0);
        let c: ValueRef = &a + &b;
        assert_eq!(c.inner.borrow().data, 3.0);
    }

    #[test]
    fn test_valueref_sum_backward() {
        let a: ValueRef = Value::new(1.0);
        let b: ValueRef = Value::new(2.0);
        let c: ValueRef = &a + &b;

        c.set_grad(1.0);

        (c.inner.borrow_mut().backward)();

        assert_eq!(a.inner.borrow().grad, 1.0);
        assert_eq!(b.inner.borrow().grad, 1.0);
    }

    #[test]
    fn test_valueref_sum_of_equal_values_backward() {
        let a: ValueRef = Value::new(1.0);
        let c: ValueRef = &a + &a;

        c.set_grad(1.0);

        (c.inner.borrow_mut().backward)();

        assert_eq!(a.inner.borrow().grad, 2.0);
    }

    //#[test]
    //fn test_value_sum_with_float() {
    //    let a = Value::new(1.0);
    //    let b = &a + 2.0;
    //    assert_eq!(b.data, 3.0);
    //    let c = 2.0 + &a;
    //    assert_eq!(c.data, 3.0);
    //}
    //
    //#[test]
    //fn test_value_mul() {
    //    let a = Value::new(2.0);
    //    let b = Value::new(3.0);
    //    let c = &a * &b;
    //    assert_eq!(c.data, 6.0);
    //}
    //
    //#[test]
    //fn test_value_mul_with_float() {
    //    let a = Value::new(3.0);
    //    let b = &a * 2.0;
    //    assert_eq!(b.data, 6.0);
    //    let c = 2.0 * &a;
    //    assert_eq!(c.data, 6.0);
    //}
}
