module.exports = {
  content: ["./web/static/index.html", "./web/static/app.js"],
  theme: {
    extend: {
      colors: {
        air: "#eaf5ff",
        mist: "#d8ecff",
        glass: "#f7fbff",
        blueLine: "#b9dcf6",
        ocean: "#2f94de",
        oceanDeep: "#1477c9",
        navy: "#233f73",
        navyDeep: "#162f5d",
        navySoft: "#e7eef9",
        graphite: "#1d2935",
        mutedBlue: "#6f879e",
        danger: "#ff4e72"
      },
      boxShadow: {
        soft: "0 18px 50px rgba(74, 103, 132, 0.16)",
        float: "0 24px 70px rgba(51, 106, 151, 0.18)",
        insetSoft: "inset 0 1px 0 rgba(255,255,255,0.9)"
      }
    }
  },
  plugins: []
};
