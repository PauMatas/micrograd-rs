use std::collections::HashSet;
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
    _backward: Box<dyn FnMut(f64)>,
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
                _backward: Box::new(|_: f64| {}),
            }))
        }
    }
}

impl ValueRef {
    pub fn set_grad(&self, new_grad: f64) {
        self.inner.borrow_mut().grad = new_grad
    }

    pub fn add_to_grad(&self, sum_to_grad: f64) {
        let g = {
            self.inner.borrow().grad
        };
        self.set_grad(g + sum_to_grad);
    }

    pub fn get_parents_topo_sort(&self) -> Vec<ValueRef> {
        let mut topo: Vec<ValueRef> = Vec::new();
        let mut visited: HashSet<*const RefCell<Value>> = HashSet::new();
        
        build_topo(self, &mut visited, &mut topo);

        topo
    }

    pub fn backward(&self) {
        self.clear_grads();
        self.set_grad(1.0);
        let topo: Vec<ValueRef> = self.get_parents_topo_sort();
        for value_ref in topo.into_iter().rev() {
            let g = {
                value_ref.inner.borrow().grad
            };
            (value_ref.inner.borrow_mut()._backward)(g);
        }
    }

    pub fn clear_grads(&self) {
        let topo: Vec<ValueRef> = self.get_parents_topo_sort();
        for value_ref in topo {
            value_ref.set_grad(0.0);
        };
    }
}

fn build_topo(value_ref: &ValueRef, visited: &mut HashSet<*const RefCell<Value>>, topo: &mut Vec<ValueRef>) {
    let ptr = Rc::as_ptr(&value_ref.inner);
    if !visited.contains(&ptr) {
        visited.insert(ptr);
        for parent_rc in &value_ref.inner.borrow()._parents {
            let parent = ValueRef { inner: parent_rc.clone() };
            build_topo(&parent, visited, topo);
        };
        topo.push(value_ref.clone());
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
        let out_rc = Rc::new(RefCell::new(Value {
            data: self.inner.borrow().data + other.inner.borrow().data,
            grad: 0.0,
            _operation: "+",
            _parents: vec![self.inner.clone(), other.inner.clone()],
            _backward: Box::new(|_: f64| {}),
        }));

        let a = self.clone();
        let b = other.clone();

        out_rc.borrow_mut()._backward = Box::new(move |g: f64| {
            a.add_to_grad(g);
            b.add_to_grad(g);
        });
        
        ValueRef { inner: out_rc }
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
            _backward: Box::new(|_: f64| {})
        }));

        let a = self.clone();
        let b = other.clone();

        out.borrow_mut()._backward = Box::new(move |g: f64| {
            let a_data = {
                a.inner.borrow().data
            };
            let b_data = {
                b.inner.borrow().data
            };
            a.add_to_grad(b_data * g);
            b.add_to_grad(a_data * g);
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
    let x: ValueRef = Value::new(2.0);
    let w: ValueRef = Value::new(1.0);
    let b: ValueRef = Value::new(3.0);

    let y: ValueRef = &(&x * &w) + &b;

    let i = Value::new(1.5);
    let j = Value::new(4.0);

    let k = &(&i * &j) + &(&x * &x);

    let result = &k * &y;

    result.backward();

    println!("x: {}, w: {}", x, w);
    println!("{:?}", result);
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

        c.backward();

        assert_eq!(a.inner.borrow().grad, 1.0);
        assert_eq!(b.inner.borrow().grad, 1.0);
    }

    #[test]
    fn test_valueref_sum_of_equal_values_backward() {
        let a: ValueRef = Value::new(1.0);
        let c: ValueRef = &a + &a;

        c.backward();

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

        c.backward();

        assert_eq!(a.inner.borrow().grad, 2.0);
        assert_eq!(b.inner.borrow().grad, 1.0);
    }

    #[test]
    fn test_valueref_mul_of_equal_values_backward() {
        let a: ValueRef = Value::new(3.0);
        let c: ValueRef = &a * &a;

        c.backward();

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

    #[test]
    fn test_valueref_clear_grads() {
        let a: ValueRef = Value::new(1.0);
        let b: ValueRef = Value::new(2.0);
        let c: ValueRef = &(&a + &b) * &a;

        c.clear_grads();

        assert_eq!(a.inner.borrow().grad, 0.0);
        assert_eq!(b.inner.borrow().grad, 0.0);
        assert_eq!(c.inner.borrow().grad, 0.0);
    }
}
