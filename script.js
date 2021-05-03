//training function
async function gaber(){
    //button disabling when training
    const button = document.querySelector('button')
    button.disabled = true;
    //get and parse training data from input forms
    let xarray = document.getElementById("xvalues").value.replace(/\s/g, '').split`,`.map(x=>+x);
    let yarray = document.getElementById("yvalues").value.replace(/\s/g, '').split`,`.map(x=>+x);
    if(xarray.length!=yarray.length){
        document.getElementById('out').innerText = "Data values don't match up!";
        return
    }
    //get the amount of epochs from form and turn it into an int
    let isepocs = document.getElementById("epochs").value;
    let domestic = parseInt(isepocs);
    //create the AI
    const sumk = tf.sequential();
    sumk.add(tf.layers.dense({units: 1, inputShape:[1]}));
    sumk.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });
    //convert data to tensors
    const xs = tf.tensor2d(xarray,[xarray.length,1]);
    const ys = tf.tensor2d(yarray,[yarray.length,1]);
    //train AI
    await sumk.fit(xs,ys, {epochs:domestic});
    //save AI params to local storage
    await sumk.save('localstorage://mothel');
    // re-enable button when done training
    button.disabled = false;
    //output text when done training
    document.getElementById('out').innerText = "Done Training!";
}
//prediction function
async function thredict(){
    //get input from form and turn it into an int
    let input = document.getElementById("in").value;
    let intl = parseInt(input);
    //load model from local storage
    const sumk = await tf.loadLayersModel('localstorage://mothel');
    //process and output the prediction
    document.getElementById('out').innerText = sumk.predict(tf.tensor2d([intl],[1,1])).arraySync();

}