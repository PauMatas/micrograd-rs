use std::fmt::Debug;
use std::fmt::Display;
use std::ops::Add;
use std::ops::Mul;
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
        {
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
                value.backward = Box::new(move || {
                    if Rc::ptr_eq(&parents[0], &parents[1]) {
                        let mut p = parents[0].borrow_mut();
                        p.grad += 2.0 * p.data * new_grad
                    } else {
                        let mut p0 = parents[0].borrow_mut();
                        let mut p1 = parents[1].borrow_mut();

                        p0.grad += p1.data * new_grad;
                        p1.grad += p0.data * new_grad;
                    }
                })
            } else {
                todo!()
            }
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

impl Add<f64> for &ValueRef {
    type Output = ValueRef;

    fn add(self, other: f64) -> ValueRef {
        let other_value_ref: ValueRef = Value::new(other);
        self + &other_value_ref
    }
}

impl Add<&ValueRef> for f64 {
    type Output = ValueRef;

    fn add(self, other: &ValueRef) -> ValueRef {
        other + self
    }
}

impl Mul for &ValueRef {
    type Output = ValueRef;

    fn mul(self, other: &ValueRef) -> ValueRef {
        let out = Rc::new(RefCell::new(Value {
            data: self.inner.borrow().data * other.inner.borrow().data,
            grad: 0.0,
            _parents: vec![self.inner.clone(), other.inner.clone()],
            _operation: "*",
            backward: Box::new(|| {})
        }));

        let a = self.clone();
        let b = other.clone();
        let out_rc = out.clone();

        out.borrow_mut().backward = Box::new(move || {
            let g = out_rc.borrow().grad;

            let mut a_borrowed = a.inner.borrow_mut();
            let mut b_borrowed = b.inner.borrow_mut();

            a_borrowed.grad += b_borrowed.data * g;
            b_borrowed.grad += a_borrowed.data * g;
        });
        
        ValueRef { inner: out }
    }
}

impl Mul<f64> for &ValueRef {
    type Output = ValueRef;

    fn mul(self, other: f64) -> ValueRef {
        let other_value_ref: ValueRef = Value::new(other);
        self * &other_value_ref
    }
}

impl Mul<&ValueRef> for f64 {
    type Output = ValueRef;

    fn mul(self, other: &ValueRef) -> ValueRef {
        other * self
    }
}

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

    #[test]
    fn test_value_sum_with_float() {
        let a = Value::new(1.0);
        let b = &a + 2.0;
        assert_eq!(b.inner.borrow().data, 3.0);
        let c = 2.0 + &a;
        assert_eq!(c.inner.borrow().data, 3.0);
    }

    #[test]
    fn test_value_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = &a * &b;
        assert_eq!(c.inner.borrow().data, 6.0);
    }

    #[test]
    fn test_valueref_mul_backward() {
        let a: ValueRef = Value::new(1.0);
        let b: ValueRef = Value::new(2.0);
        let c: ValueRef = &a * &b;

        c.set_grad(1.0);

        (c.inner.borrow_mut().backward)();

        assert_eq!(a.inner.borrow().grad, 2.0);
        assert_eq!(b.inner.borrow().grad, 1.0);
    }

    #[test]
    fn test_valueref_mul_of_equal_values_backward() {
        let a: ValueRef = Value::new(3.0);
        let c: ValueRef = &a * &a;

        c.set_grad(1.0);

        (c.inner.borrow_mut().backward)();

        assert_eq!(a.inner.borrow().grad, 6.0);
    }

    #[test]
    fn test_value_mul_with_float() {
        let a = Value::new(3.0);
        let b = &a * 2.0;
        assert_eq!(b.inner.borrow().data, 6.0);
        let c = 2.0 * &a;
        assert_eq!(c.inner.borrow().data, 6.0);
    }
}
