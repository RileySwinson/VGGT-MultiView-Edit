var armIds = ["unedited", "stock", "scene", "targets"];

var scenes = {
    bear: {
        prompts: ["grizzly", "polar", "golden"],
        text: [
            "<em>\"Turn the bear into a Grizzly bear\"</em>",
            "<em>\"Turn the bear into a Polar bear\"</em>",
            "<em>\"Turn the bear into a golden statue\"</em>",
        ],
    },
    farm: {
        prompts: ["autumn", "watercolor"],
        text: [
            "<em>\"Make it autumn\"</em>",
            "<em>\"Make the scene look as if it's painted in watercolor\"</em>",
        ],
    },
    face: {
        prompts: ["clown", "marble", "animation"],
        text: [
            "<em>\"Make him look like a clown\"</em>",
            "<em>\"Make his face look like a marble sculpture\"</em>",
            "<em>\"Make him look like a stylized animation character\"</em>",
        ],
    },
};

var ablationIds = ["targets", "abl_no_fit", "abl_no_prepass", "abl_vggt_origin", "scene"];

function switchVideo(video, src) {
    var time = video.currentTime || 0;
    video.src = src;
    video.addEventListener("loadedmetadata", function seek() {
        video.removeEventListener("loadedmetadata", seek);
        video.currentTime = time % video.duration;
        video.play();
    });
    video.load();
}

function setup(name) {
    var scene = scenes[name];
    scene.prompt = 0;
    scene.arm = 3;
    scene.video = document.getElementById(name + "-video");
    scene.caption = document.getElementById(name + "-text");
    scene.promptButtons = scene.prompts.map(function (p) { return document.getElementById(name + "-" + p); });
    scene.armButtons = armIds.map(function (a) { return document.getElementById(name + "-" + a); });
    scene.promptButtons.forEach(function (b, i) { b.addEventListener("click", function () { scene.prompt = i; show(name); }); });
    scene.armButtons.forEach(function (b, i) { b.addEventListener("click", function () { scene.arm = i; show(name); }); });
    show(name);
}

function show(name) {
    var scene = scenes[name];
    scene.promptButtons.forEach(function (b, i) { b.classList.toggle("active-btn", i == scene.prompt); });
    scene.armButtons.forEach(function (b, i) { b.classList.toggle("active-btn", i == scene.arm); });
    scene.caption.innerHTML = scene.text[scene.prompt];
    switchVideo(scene.video, "data/videos/" + name + "/" + scene.prompts[scene.prompt] + "-" + armIds[scene.arm] + ".mp4");
}

function setupAblation() {
    var video = document.getElementById("ablation-video");
    var buttons = ablationIds.map(function (a) { return document.getElementById("ablation-" + a); });
    function showAblation(i) {
        buttons.forEach(function (b, j) { b.classList.toggle("active-btn", j == i); });
        switchVideo(video, "data/videos/ablation/" + ablationIds[i] + ".mp4");
    }
    buttons.forEach(function (b, i) { b.addEventListener("click", function () { showAblation(i); }); });
    showAblation(0);
}

$(function () {
    setup("bear");
    setup("farm");
    setup("face");
    setupAblation();
});
