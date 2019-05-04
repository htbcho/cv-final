$(document).ready(function(){
    const label = '';
    const word_bank = ["CAT", "DOG", "YAY", "VISION", "FUN", "DAD", "EYE"];
    // const word_bank = ["L"];
    let word;
    let all_letters;
    // let curr_letter;
    let curr_index;

    // document.getElementById("new-word").addEventListener("onclick", newWord());
    var button = document.getElementById("new-word");
    button.onclick = function() {
        var container = $("#word-container");
        container.empty();
        var max = word_bank.length-1;
        var min = 0;
        var index = Math.floor(Math.random()*(max-min+1)+min);
        word = word_bank[index];

        // var l = document.getElementById("word");
        // l.innerText = word;
        all_letters = [];
        // var container = document.getElementById("word-container");
        var container = $("#word-container");
        container.empty();
        curr_index = 0;
        for (var i=0; i<word.length; i++) {
            console.log("loop");
            var letter = document.createElement("p");
            letter.innerText = word.charAt(i);
            letter.style.fontSize = "30px";
            letter.style.fontWeight = "bold";
            letter.style.fontFamily = "'Fredoka One', cursive";
            container.append(letter);
            all_letters.push(letter);
        }
        console.log("after loop " + container.children());
        
        // l.style.marginTop = "0px";
        // l.style.paddingBottom = "10px";
        curr_index = 0;
        // curr_letter = word.charAt(0);
        // l.textContent.charAt(0).fontcolor = "#42a4f4";
        // console.log(l.textContent.charAt(0).fontcolor(getColor()));
        // console.log('asdada');
    }
    


    function getImage() {
        const video = document.querySelector("#webcamElement");
        var canvas = document.getElementById("canvas"); //if this doesn't work, try videoElement (previously container)
        var ctx = canvas.getContext('2d');
        video.addEventListener('play', draw(video, canvas, ctx, .0001));
        //set width and height of mirror?????
    }
    //https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/index.js

    //before was a var
    function draw(video, canvas, context, frameRate) {
        // console.log("draw");
        // var topcanvas = document.getElementById("webcam-interaction");
        var ret = context.drawImage(video, 0, 0, canvas.width, canvas.height);
        var url = canvas.toDataURL("image/jpeg"); //.replace("image/jpeg", "image/octet-stream")
        var image = document.getElementById("mirror");
        image.src = url;

        getLabel(image);


        setTimeout(draw, 1/frameRate, video, canvas, context, frameRate);
        //maybe push img to model here?
    }

    function getLabel(image) { //eventually pass in letter <some set up stuff: https://stackoverflow.com/questions/11071100/jquery-uncaught-typeerror-illegal-invocation-at-ajax-request-several-eleme>
        $.post( "/model", {
            url: image.src,
            crossDomain: true,
            processData: false,
            // success: function(data) {
            //     console.log('what');
            //     console.log(data);
            //     // console.log(response);
            //     // 
            }, 
            function(err, req, response){ 
                console.log(response["responseJSON"]["label"]);
                var label = response["responseJSON"]["label"]; 
                changeLetter(label);
            });
    }

    function changeLetter(label) {
        if (label === word.charAt(curr_index)) {
            var ele = all_letters[curr_index];
            ele.style.color = "#1fa851";
            console.log(curr_index);
            console.log(ele.style.color);
            console.log("new letter " + ele.innerText);
            curr_index++;
        }
    }


    getImage();
});