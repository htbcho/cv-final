$(document).ready(function(){
    const label = '';


    function getImage() {
        const video = document.querySelector("#webcamElement");
        var canvas = document.getElementById("canvas"); //if this doesn't work, try videoElement (previously container)
        var ctx = canvas.getContext('2d');
        video.addEventListener('play', draw(video, canvas, ctx, 1));
        //set width and height of mirror?????
    }
    //https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/index.js

    //before was a var
    function draw(video, canvas, context, frameRate) {
        // console.log("draw");
        // var topcanvas = document.getElementById("webcam-interaction");
        var ret = context.drawImage(video, 0, 0, canvas.width, canvas.height);
        var url = canvas.toDataURL("image/jpeg").replace("image/jpeg", "image/octet-stream");
        var image = document.getElementById("mirror");
        image.src = url;
        //ok nowwww i think this will be ready for the model
        getLabel(image);
        // var label = callbackFunc();


        setTimeout(draw, 1/frameRate, video, canvas, context, frameRate);
        //maybe push img to model here?
    }

    function getLabel(image) { //eventually pass in letter <some set up stuff: https://stackoverflow.com/questions/11071100/jquery-uncaught-typeerror-illegal-invocation-at-ajax-request-several-eleme>
        $.post( "/model", {
            // beforeSend: function(xhrObj){
            //     // xhrObj.setRequestHeader("Content-Type","application/json");
            //     // xhrObj.setRequestHeader("Accept","application/json");
            //     // xhrObj.setRequestHeader("Access-Control-Allow-Origin", "*");
            //     xhrObj.setRequestHeader("Access-Control-Allow-Headers","x-requested-with");
            // },
            // method: 'POST',
            // image: JSON.stringify('test'),
            // image: image,
            url: image.src,
            crossDomain: true,
            processData: false,
            // success: function(data) {
            //     console.log('what');
            //     console.log(data);
            //     // console.log(response);
            //     // elem.innerHTML = TURN THE LETTER GREEN
            }, 
            function(err, req, response){ 
                console.log(response["responseJSON"]["label"]); 
            });
    }

// function callbackFunc(response) {
//     label = response;
// }

    getImage();
});