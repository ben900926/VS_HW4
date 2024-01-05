const videoContainer = document.getElementById('video-container');

function mousepos(event) {
    var x = (event.pageX - videoContainer.offsetLeft) / videoContainer.clientWidth;
    var y = (event.pageY - videoContainer.offsetTop) / videoContainer.clientHeight;
    if (0 <= x && x <= 1 && 0 <= y && y <= 1){
        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/data', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify({x: x, y: y}));
    }
  }

  window.addEventListener('mousedown', mousepos);
  