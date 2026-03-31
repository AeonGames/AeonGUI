function setStatus(text) {
  document.getElementById('status').textContent = text;
}

function btnMouseOver(btn, hoverGrad, label) {
  btn.querySelector('rect').setAttribute('fill', hoverGrad);
  setStatus('Mouse over: ' + label);
}

function btnMouseOut(btn, normalGrad, label) {
  btn.querySelector('rect').setAttribute('fill', normalGrad);
  btn.setAttribute('filter', 'url(#shadow)');
  btn.setAttribute('transform', '');
  setStatus('Mouse left: ' + label);
}

function btnMouseDown(btn, activeGrad, label) {
  btn.querySelector('rect').setAttribute('fill', activeGrad);
  btn.setAttribute('filter', 'url(#shadowPressed)');
  btn.setAttribute('transform', 'translate(1,1)');
  setStatus('Mouse down: ' + label);
}

function btnMouseUp(btn, hoverGrad, label) {
  btn.querySelector('rect').setAttribute('fill', hoverGrad);
  btn.setAttribute('filter', 'url(#shadow)');
  btn.setAttribute('transform', '');
  setStatus('Clicked: ' + label + '!');
}

function toggleMouseDown(btn) {
  var r = btn.querySelector('rect');
  var t = btn.querySelector('text');
  if (r.getAttribute('fill') === '#888') {
    r.setAttribute('fill', '#4a4');
    r.setAttribute('stroke', '#2a2');
    t.textContent = 'ON';
    setStatus('Toggle: ON');
  } else {
    r.setAttribute('fill', '#888');
    r.setAttribute('stroke', '#666');
    t.textContent = 'OFF';
    setStatus('Toggle: OFF');
  }
}
