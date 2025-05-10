use micrograd_rs::ValueRef;

fn main() {
    let x = ValueRef::new(2.0);
    let w = ValueRef::new(1.0);
    let b = ValueRef::new(3.0);

    let y: ValueRef = &(&x * &w) + &b;

    let i = ValueRef::new(1.5);
    let j = ValueRef::new(4.0);

    let k = &(&i * &j) + &(&x * &x);

    let result = &k * &y;

    result.backward();

    println!("x: {}, w: {}", x, w);
    println!("{:?}", result);
}

