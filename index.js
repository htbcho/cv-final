// import * as tf from '@tensorflow/tfjs';
// import Webcam from './webcam.js';

const label = '';
// const webcam = new Webcam(document.getElementById('webcamElement'));

function getImage() {
    const video = document.querySelector("#webcamElement");
    var canvas = document.getElementById("canvas"); //if this doesn't work, try videoElement (previously container)
    var ctx = canvas.getContext('2d');
    video.addEventListener('play', draw(video, canvas, ctx, 1));
}
// var capture = function() {
//     const video = document.querySelector("#webcamElement");
//     var canvas = document.getElementById("canvas"); //if this doesn't work, try videoElement (previously container)
//     var ctx = canvas.getContext('2d');
//     video.addEventListener('play', draw(video, canvas, ctx, 3));
// }

var draw = function(video, canvas, context, frameRate) {
    // console.log("draw");
    var topcanvas = document.getElementById("webcam-interaction");
    var ret = context.drawImage(video, 0, 0, canvas.width, canvas.height);
    var url = canvas.toDataURL("image/jpeg").replace("image/jpeg", "image/octet-stream");
    var image = new Image();
    image.src = url;
    //ok nowwww i think this will be ready for the model

    getLabel(image);
    var label = callbackFunc();


    setTimeout(draw, 1/frameRate, video, canvas, context, frameRate);
    //maybe push img to model here?
}

function getLabel(image) {
    $.ajax({
        type: 'POST',
        url: 'load_test_model.py', //maybe no backlash
        data: { param: image},
        success: callbackFunc
    });
}

function callbackFunc(response) {
    label = response;
}

getImage();