window.dash_clientside = Object.assign({}, window.dash_clientside, {
  netpyneModeler: {
    applyTheme: function (darkEnabled) {
      const isDark = !!darkEnabled;
      document.documentElement.setAttribute("data-bs-theme", isDark ? "dark" : "light");
      document.body.classList.toggle("theme-dark-body", isDark);
      document.body.classList.toggle("theme-light-body", !isDark);
      return isDark ? "dark" : "light";
    },
    resetToDefaults: function (nClicks) {
      if (!nClicks) {
        return window.dash_clientside.no_update;
      }
      window.setTimeout(function () {
        window.location.reload();
      }, 0);
      return window.dash_clientside.no_update;
    }
  }
});
