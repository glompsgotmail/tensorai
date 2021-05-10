

/**
 * We want to learn the coefficients that give correct solutions to the
 * following cubic equation:
 *      y = a * x^3 + b * x^2 + c * x + d
 * In other words we want to learn values for:
 *      a
 *      b
 *      c
 *      d
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of 'xs' and 'ys' to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 */

// Step 1. Set up variables, these are the things we want the model
// to learn in order to do prediction accurately. We will initialize
// them with random values.
let a = tf.variable(tf.scalar(Math.random()));
let b = tf.variable(tf.scalar(Math.random()));
let c = tf.variable(tf.scalar(Math.random()));
let d = tf.variable(tf.scalar(Math.random()));
const learningRate = 0.0001;
    const optimizer = tf.train.sgd(learningRate);

// Step 2. Create an optimizer, we will use this later. You can play
// with some of these values to see how the model performs.


// Step 3. Write our training process functions.

/*
 * This function represents our 'model'. Given an input 'x' it will try and
 * predict the appropriate output 'y'.
 *
 * It is also sometimes referred to as the 'forward' step of our training
 * process. Though we will use the same function for predictions later.
 *
 * @return number predicted y value
 */
function predict(x) {
  // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => {
    return a.mul(x.pow(tf.scalar(3, 'int32')))
      .add(b.mul(x.square()))
      .add(c.mul(x))
      .add(d);
  });
}

/*
 * This will tell us how good the 'prediction' is given what we actually
 * expected.
 *
 * prediction is a tensor with our predicted y values.
 * labels is a tensor with the y values the model should have predicted.
 */
function loss(prediction, labels) {
  // Having a good error function is key for training a machine learning model
  const error = prediction.sub(labels).square().mean();
  return error;
}

/*
 * This will iteratively train our model.
 *
 * xs - training data x values
 * ys — training data y values
 */
async function train(xs, ys, numIterations) {
    const learningRate = 0.0025;
    const optimizer = tf.train.sgd(learningRate);
    let start = new Date();
    let x = new Date();
  for (let iter = 0; iter < numIterations; iter++) {
    x = new Date();
    document.getElementById("out").innerText = Math.round(100*(iter/numIterations)) + "%" + "\n" + iter + " of " + numIterations + " generations." + "\n" + Math.round(iter/(Math.round((x-start)/1000)))+ " generations/sec.";
    optimizer.minimize(() => {
      // Feed the examples into the model
      for(let i = 0; i<xs.length;i++){
      const pred = predict(xs[i]);
      return loss(pred, ys[i]);
    }
    });

    // Use tf.nextFrame to not block the browser.
    await tf.nextFrame();
  }
}

async function learn() {
    const btn = document.querySelector("button");
    btn.disabled=true;
    let xarray = document.getElementById("xvalues").value.replace(/\s/g, '').split`,`.map(x=>+x).map(x=>tf.tensor(x,[1,1]));
    let yarray = document.getElementById("yvalues").value.replace(/\s/g, '').split`,`.map(x=>+x).map(x=>tf.tensor(x,[1,1]));
    if(xarray.length!=yarray.length){
        document.getElementById('out').innerText = "Data values don't match up!";
        return
    }
    let isepocs = document.getElementById("epochs").value;
    let domestic = parseInt(isepocs);
  // Train the model!
    await train(xarray, yarray, domestic);
    btn.disabled=false;
    document.getElementById("out").innerText = "Done Training!";

}

async function thredict(){
    document.getElementById("out").innerText = predict(tf.tensor(parseInt(document.getElementById("in").value))).dataSync();
    console.log(""+a.dataSync()+"\n"+b.dataSync()+"\n"+c.dataSync()+"\n"+d.dataSync());
}
