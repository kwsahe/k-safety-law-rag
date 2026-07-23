(() => {
  const savedTheme = localStorage.getItem("k-safety-theme");
  const preferredTheme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = savedTheme || preferredTheme;
})();
