use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub, Div, Neg};
use std::{rc::Rc, cell::RefCell, vec, collections::HashSet};

struct Value {
    data: f64,
    grad: f64,
    parents: Vec<Rc<RefCell<Value>>>,
    operation: &'static str,
    backward: Box<dyn FnMut(f64)>,
}

impl PartialEq for Value {
    fn eq(&self, rhs: &Self) -> bool {
        self.data == rhs.data
    }
}

#[derive(Clone)]
pub struct ValueRef {
    inner: Rc<RefCell<Value>>,
}

impl Value {
    fn new(data: f64) -> ValueRef {
        ValueRef {
            inner: Rc::new(RefCell::new(Value {
                data,
                grad: 0.0,
                operation: "",
                parents: vec![],
                backward: Box::new(|_: f64| {}),
            }))
        }
    }
}

impl ValueRef {
    pub fn new(data: f64) -> ValueRef {
        Value::new(data)
    }

    pub fn set_grad(&self, new_grad: f64) {
        self.inner.borrow_mut().grad = new_grad
    }

    fn add_to_grad(&self, sum_to_grad: f64) {
        let g = {
            self.inner.borrow().grad
        };
        self.set_grad(g + sum_to_grad);
    }

    fn getparents_topo_sort(&self) -> Vec<ValueRef> {
        let mut topo: Vec<ValueRef> = Vec::new();
        let mut visited: HashSet<*const RefCell<Value>> = HashSet::new();
        
        build_topo(self, &mut visited, &mut topo);

        topo
    }

    pub fn backward(&self) {
        self.clear_grads();
        self.set_grad(1.0);
        let topo: Vec<ValueRef> = self.getparents_topo_sort();
        for value_ref in topo.into_iter().rev() {
            let g = {
                value_ref.inner.borrow().grad
            };
            (value_ref.inner.borrow_mut().backward)(g);
        }
    }

    pub fn clear_grads(&self) {
        let topo: Vec<ValueRef> = self.getparents_topo_sort();
        for value_ref in topo {
            value_ref.set_grad(0.0);
        };
    }

    pub fn pow(&self, exp: f64) -> ValueRef {
        ValueRef::new(self.inner.borrow().data.powf(exp))
    }

    pub fn relu(&self) -> ValueRef {
        let self_data = self.inner.borrow().data;
        let out_rc = Rc::new(RefCell::new(Value {
            data: if self_data > 0.0 { self_data } else { 0.0 },
            grad: 0.0,
            operation: "ReLU",
            parents: vec![self.inner.clone()],
            backward: Box::new(|_: f64| {}),
        }));

        let a = self.clone();

        out_rc.borrow_mut().backward = Box::new(move |g: f64| {
            a.add_to_grad(if self_data > 0.0 { g } else { 0.0 });
        });
        
        ValueRef { inner: out_rc }
    }
}

impl From<f64> for ValueRef {
    fn from(x: f64) -> ValueRef {
        Value::new(x)
    }
}

fn build_topo(value_ref: &ValueRef, visited: &mut HashSet<*const RefCell<Value>>, topo: &mut Vec<ValueRef>) {
    let ptr = Rc::as_ptr(&value_ref.inner);
    if !visited.contains(&ptr) {
        visited.insert(ptr);
        for parent_rc in &value_ref.inner.borrow().parents {
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
            self.data, self.grad, self.operation
        )?;

        if !self.parents.is_empty() {
            write!(f, ", parents: [")?;
            for parent_rc in self.parents.iter() {
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

impl Neg for &ValueRef {
    type Output = ValueRef;
    
    fn neg(self) -> Self::Output {
        ValueRef::new(-self.inner.borrow().data)
    }
}

impl Add for &ValueRef {
    type Output = ValueRef;

    fn add(self, rhs: Self) -> Self::Output {
        let out_rc = Rc::new(RefCell::new(Value {
            data: self.inner.borrow().data + rhs.inner.borrow().data,
            grad: 0.0,
            operation: "+",
            parents: vec![self.inner.clone(), rhs.inner.clone()],
            backward: Box::new(|_: f64| {}),
        }));

        let a = self.clone();
        let b = rhs.clone();

        out_rc.borrow_mut().backward = Box::new(move |g: f64| {
            a.add_to_grad(g);
            b.add_to_grad(g);
        });
        
        ValueRef { inner: out_rc }
    }
}

impl Add<f64> for &ValueRef {
    type Output = ValueRef;

    fn add(self, rhs: f64) -> Self::Output {
        let rhs_value_ref: ValueRef = Value::new(rhs);
        self + &rhs_value_ref
    }
}

impl Add<&ValueRef> for f64 {
    type Output = ValueRef;

    fn add(self, rhs: &ValueRef) -> Self::Output {
        rhs + self
    }
}

impl Sub for &ValueRef {
    type Output = ValueRef;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &(-rhs)
    }
}

impl Sub<f64> for &ValueRef {
    type Output = ValueRef;

    fn sub(self, rhs: f64) -> Self::Output {
        let rhs_value_ref: ValueRef = Value::new(rhs);
        self - &rhs_value_ref
    }
}

impl Sub<&ValueRef> for f64 {
    type Output = ValueRef;

    fn sub(self, rhs: &ValueRef) -> Self::Output {
        let self_value_ref: ValueRef = Value::new(self);
        &self_value_ref - rhs
    }
}

impl Mul for &ValueRef {
    type Output = ValueRef;

    fn mul(self, rhs: Self) -> Self::Output {
        let out = Rc::new(RefCell::new(Value {
            data: self.inner.borrow().data * rhs.inner.borrow().data,
            grad: 0.0,
            parents: vec![self.inner.clone(), rhs.inner.clone()],
            operation: "*",
            backward: Box::new(|_: f64| {})
        }));

        let a = self.clone();
        let b = rhs.clone();

        out.borrow_mut().backward = Box::new(move |g: f64| {
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

    fn mul(self, rhs: f64) -> ValueRef {
        let rhs_value_ref: ValueRef = Value::new(rhs);
        self * &rhs_value_ref
    }
}

impl Mul<&ValueRef> for f64 {
    type Output = ValueRef;

    fn mul(self, rhs: &ValueRef) -> Self::Output {
        rhs * self
    }
}

impl Div for &ValueRef {
    type Output = ValueRef;

    fn div(self, rhs: Self) -> Self::Output {
        self * &(rhs.pow(-1.0))
    }
}

impl Div<f64> for &ValueRef {
    type Output = ValueRef;

    fn div(self, rhs: f64) -> ValueRef {
        let rhs_value_ref: ValueRef = Value::new(rhs);
        self / &rhs_value_ref
    }
}

impl Div<&ValueRef> for f64 {
    type Output = ValueRef;

    fn div(self, rhs: &ValueRef) -> Self::Output {
        let self_value_ref: ValueRef = Value::new(self);
        &self_value_ref / rhs
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data() {
        let x: ValueRef = Value::new(3.14);
        assert_eq!(x.inner.borrow().data, 3.14);
    }

    #[test]
    fn test_sum() {
        let a: ValueRef = Value::new(1.0);
        let b: ValueRef = Value::new(2.0);
        let c: ValueRef = &a + &b;
        assert_eq!(c.inner.borrow().data, 3.0);
    }

    #[test]
    fn test_sum_backward() {
        let a: ValueRef = Value::new(1.0);
        let b: ValueRef = Value::new(2.0);
        let c: ValueRef = &a + &b;

        c.backward();

        assert_eq!(a.inner.borrow().grad, 1.0);
        assert_eq!(b.inner.borrow().grad, 1.0);
    }

    #[test]
    fn test_sum_of_equal_values_backward() {
        let a: ValueRef = Value::new(1.0);
        let c: ValueRef = &a + &a;

        c.backward();

        assert_eq!(a.inner.borrow().grad, 2.0);
    }

    #[test]
    fn test_sum_with_float() {
        let a = Value::new(1.0);
        let b = &a + 2.0;
        assert_eq!(b.inner.borrow().data, 3.0);
        let c = 2.0 + &a;
        assert_eq!(c.inner.borrow().data, 3.0);
    }

    #[test]
    fn test_mul() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = &a * &b;
        assert_eq!(c.inner.borrow().data, 6.0);
    }

    #[test]
    fn test_mul_backward() {
        let a: ValueRef = Value::new(1.0);
        let b: ValueRef = Value::new(2.0);
        let c: ValueRef = &a * &b;

        c.backward();

        assert_eq!(a.inner.borrow().grad, 2.0);
        assert_eq!(b.inner.borrow().grad, 1.0);
    }

    #[test]
    fn test_mul_of_equal_values_backward() {
        let a: ValueRef = Value::new(3.0);
        let c: ValueRef = &a * &a;

        c.backward();

        assert_eq!(a.inner.borrow().grad, 6.0);
    }

    #[test]
    fn test_mul_with_float() {
        let a = Value::new(3.0);
        let b = &a * 2.0;
        assert_eq!(b.inner.borrow().data, 6.0);
        let c = 2.0 * &a;
        assert_eq!(c.inner.borrow().data, 6.0);
    }

    #[test]
    fn test_clear_grads() {
        let a: ValueRef = Value::new(1.0);
        let b: ValueRef = Value::new(2.0);
        let c: ValueRef = &(&a + &b) * &a;

        c.clear_grads();

        assert_eq!(a.inner.borrow().grad, 0.0);
        assert_eq!(b.inner.borrow().grad, 0.0);
        assert_eq!(c.inner.borrow().grad, 0.0);
    }

    #[test]
    fn test_sub() {
        let a = ValueRef::new(3.0);
        let b = &a - 2.0;
        assert_eq!(b.inner.borrow().data, 1.0);
        let c = 2.0 - &a;
        assert_eq!(c.inner.borrow().data, -1.0);
        let d = &a - &a;
        assert_eq!(d.inner.borrow().data, 0.0);
    }

    #[test]
    fn test_div() {
        let a = ValueRef::new(3.0);
        let b = &a / 2.0;
        assert_eq!(b.inner.borrow().data, 1.5);
        let c = 6.0 / &a;
        assert_eq!(c.inner.borrow().data, 2.0);
        let d = &a / &a;
        assert_eq!(d.inner.borrow().data, 1.0);
    }

    #[test]
    fn test_pow() {
        let a = ValueRef::new(2.0);
        let b = a.pow(3.0);
        assert_eq!(b.inner.borrow().data, 8.0);
    }

    #[test]
    fn test_relu() {
        let a = ValueRef::new(2.0);
        let b = a.relu();
        assert_eq!(b.inner.borrow().data, 2.0);
        let c = (-&a).relu();
        assert_eq!(c.inner.borrow().data, 0.0);
    }

    #[test]
    fn test_relu_backward() {
        let a = ValueRef::new(2.0);
        let b = a.relu();
        b.backward();
        assert_eq!(a.inner.borrow().grad, 1.0);

        let c = ValueRef::new(-2.0);
        let d = c.relu();
        d.backward();
        assert_eq!(c.inner.borrow().grad, 0.0);
    }
}

