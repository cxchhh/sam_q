function init() {
    globalThis.updateCanvas = function () {
        var canvas = document.querySelector("#canvas");
        if (canvas) {
            canvas.remove();
        }
        inputImg = document.querySelector(".image-frame");
        if (!inputImg) {
            return null;
        }
        else {
            inputImg.style.position = 'relative'
            canvas = document.createElement('canvas');
            canvas.id = 'canvas';
            canvas.style.border = "0px solid red";
            canvas.style.position = 'absolute';
            canvas.width = inputImg.clientWidth;
            canvas.height = inputImg.clientHeight;
            inputImg.insertBefore(canvas, inputImg.lastChild);
            return canvas;
        }
    }

    globalThis.points = [];
    globalThis.box = [];

    globalThis.setSegEverything = function () {
        var canvas = globalThis.updateCanvas();
    }

    globalThis.setSegPoints = function () {
        var canvas = globalThis.updateCanvas();
        if (!canvas) {
            return;
        }
        let ctx = canvas.getContext("2d");
        var canvasBound = canvas.getBoundingClientRect();
        globalThis.points = [];
        rad = 3;
        canvas.onmousedown = function (e) {
            if (globalThis.restart) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                globalThis.points = [];
                globalThis.restart = false;
            }
            let x = e.clientX - canvasBound.x;
            let y = e.clientY - canvasBound.y;
            globalThis.points.push([x / canvas.width, y / canvas.height]);
            ctx.strokeStyle = 'red';
            ctx.lineWidth = "2px";
            ctx.beginPath();
            ctx.arc(x, y, rad, 0, 2 * Math.PI, 0);
            ctx.closePath();
            ctx.fillStyle = "white";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x, y, rad * 0.5, 0, 2 * Math.PI, 0);
            ctx.closePath();
            ctx.fillStyle = "red";
            ctx.fill();
        }
    }

    globalThis.setSegBox = function () {
        var canvas = globalThis.updateCanvas();
        if (!canvas) {
            return;
        }
        let ctx = canvas.getContext("2d");
        var canvasBound = canvas.getBoundingClientRect();
        globalThis.box = [];
        canvas.onmousedown = function (e) {
            if (globalThis.restart) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                globalThis.box = [];
                globalThis.restart = false;
            }
            if (globalThis.box.length > 0) return;
            let x = e.clientX - canvasBound.x;
            let y = e.clientY - canvasBound.y;
            globalThis.box.push(x / canvas.width, y / canvas.height);
        }
        canvas.onmousemove = function (e) {
            if (globalThis.box.length == 0 || globalThis.box.length == 4) return;
            let stx = globalThis.box[0] * canvas.width;
            let sty = globalThis.box[1] * canvas.height;
            let x = e.clientX - canvasBound.x;
            let y = e.clientY - canvasBound.y;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
            ctx.rect(stx, sty, x - stx, y - sty);
            ctx.closePath();
            ctx.strokeStyle = "red";
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        canvas.onmouseup = function (e) {
            if (globalThis.box.length == 0 || globalThis.box.length == 4) return;
            let x = e.clientX - canvasBound.x;
            let y = e.clientY - canvasBound.y;
            globalThis.box.push(x / canvas.width, y / canvas.height);
        }
    }

    globalThis.resetCanvas = function () {
        var canvas = globalThis.updateCanvas();
        globalThis.points = [];
        globalThis.box = [];
        globalThis.changeToMode(globalThis.mode);
    }

    globalThis.changeToMode = function (m) {
        if (typeof (m) == undefined) {
            globalThis.setSegEverything();
        }
        else if (m == 'everything') {
            globalThis.setSegEverything();
        }
        else if (m == 'points') {
            globalThis.setSegPoints();
        }
        else if (m == 'box') {
            globalThis.setSegBox();
        }

    }

    globalThis.submit = function (mt,i, m, p, mul) {
        res = [mt, i, m, null, mul];
        if (globalThis.mode == 'everything') {
            res = [mt, i, m, null, mul];
        }
        else if (globalThis.mode == 'points') {
            res = [mt, i, m, globalThis.points, mul];
        }
        else if (globalThis.mode == 'box') {
            res = [mt, i, m, globalThis.box, mul];
        }
        globalThis.restart = true;
        return res;
    }
}
