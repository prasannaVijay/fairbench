// Load Mermaid.js and initialize all .mermaid divs on each page load.
// Using the non-ESM bundle so it works without module bundling.
(function () {
  var script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
  script.onload = function () {
    mermaid.initialize({
      startOnLoad: false,
      theme:
        document.body.getAttribute("data-md-color-scheme") === "slate"
          ? "dark"
          : "default",
    });
    document.querySelectorAll(".mermaid").forEach(function (el) {
      mermaid.init(undefined, el);
    });
  };
  document.head.appendChild(script);
})();
