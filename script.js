async function gaber(){
    const button = document.querySelector('button')
    button.disabled = true;
    let isepocs = document.getElementById("epochs").value;
    let domestic = parseInt(isepocs);
    const sumk = tf.sequential();
    sumk.add(tf.layers.dense({units: 1, inputShape:[1]}));
    sumk.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });
    const xs = tf.tensor2d([-1,0,1,2,3,4],[6,1]);
    const ys = tf.tensor2d([-3,-1,1,3,5,7],[6,1]);
    await sumk.fit(xs,ys, {epochs:domestic});
    await sumk.save('localstorage://mothel');
    button.disabled = false;
    document.getElementById('out').innerText = "Done Training!";
}

async function thredict(){
    let input = document.getElementById("in").value;
    let intl = parseInt(input);
    const sumk = await tf.loadLayersModel('localstorage://mothel');
    document.getElementById('out').innerText = sumk.predict(tf.tensor2d([intl],[1,1])).arraySync();

}