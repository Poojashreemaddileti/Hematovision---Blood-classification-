const input = document.querySelector("#file-input");
const drop = document.querySelector("#dropzone");
const filename = document.querySelector("#filename");

if (input && drop) {
  ["dragenter","dragover"].forEach(evt => drop.addEventListener(evt, e => {
    e.preventDefault();
    drop.style.borderColor = "#2458ff";
    drop.style.background = "#f7f8fb";
  }));
  ["dragleave","drop"].forEach(evt => drop.addEventListener(evt, e => {
    e.preventDefault();
    drop.style.borderColor = "";
    drop.style.background = "";
  }));
  drop.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      showName(input.files[0]);
    }
  });
  input.addEventListener("change", () => {
    if (input.files.length) showName(input.files[0]);
  });
}
function showName(file) {
  if (filename) filename.textContent = `${file.name} • ${(file.size/1024/1024).toFixed(2)} MB`;
}
