$(document).ready(function(){
    const label = '';
    // const word_bank = ["CAT", "DOG", "YAY", "VISION", "FUN", "DAD", "EYE"];
    // const word_bank = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH", "II", "JJ", "KK", "LL", "MM", "NN", "OO", "PP", "QQ", "RR", "SS", "TT", "UU", "VV", "WW", "XX", "YY", "ZZ"]
    const word_bank = ["HOW"];
    let word;
    let all_letters;
    // let curr_letter;
    let curr_index;

    // document.getElementById("new-word").addEventListener("onclick", newWord());
    var button = document.getElementById("new-word");
    button.onclick = function() {
        // var container = $("#word-container");
        var wrapper = $("#center-wrapper");
        wrapper.empty();
        var max = word_bank.length-1;
        var min = 0;
        var index = Math.floor(Math.random()*(max-min+1)+min);
        word = word_bank[index];
        all_letters = [];
        var wrapper = $("#center-wrapper");
        wrapper.empty();
        curr_index = 0;
        for (var i=0; i<word.length; i++) {
            var letter_container = document.createElement("div"); //for styling purposes
            letter_container.style.margin = "0 auto";
            letter_container.style.marginLeft = "15px";
            letter_container.style.marginRight = "15px";
            letter_container.style.display = "inline-block";
            var letter = document.createElement("p");
            letter_container.appendChild(letter);
            letter.id = i.toString();
            letter.innerText = word.charAt(i);
            letter.style.fontSize = "55px";
            letter.style.fontWeight = "bold";
            letter.style.fontFamily = "'Fredoka One', cursive";
            wrapper.append(letter_container);
            all_letters.push(letter);
        }
        
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

        pushImage(image);


        setTimeout(draw, 1/frameRate, video, canvas, context, frameRate);
        //maybe push img to model here?
    }

    function pushImage(image) { //eventually pass in letter <some set up stuff: https://stackoverflow.com/questions/11071100/jquery-uncaught-typeerror-illegal-invocation-at-ajax-request-several-eleme>
        $.post( "/model", {
            url: image.src,
            crossDomain: true,
            processData: false,
            success: changeLetter()
            }, 
            // function(err, req, response){ 
            //     // getLabel();
            //     // console.log("done POST");
            //     // console.log(response["responseJSON"]["label"]);
            //     // var label = response["responseJSON"]["label"]; 
            //     // changeLetter();
            // }
            );
    }

    function changeLetter() {
        $.get("/model", function(data, status){
            var label = data.label;
            console.log("predicted: " + label);
            if (label === word.charAt(curr_index)) {
                console.log("Correct!");
                var ele = all_letters[curr_index];
                ele.style.color = "#1fa851";
                curr_index++;
            }
        });
        
    }


    getImage();
});