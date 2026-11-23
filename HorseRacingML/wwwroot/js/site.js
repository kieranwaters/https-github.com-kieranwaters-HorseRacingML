// Please see documentation at https://learn.microsoft.com/aspnet/core/client-side/bundling-and-minification
// for details on configuring this project to bundle and minify static web assets.

// Write your JavaScript code.

// Dark mode toggle. The initial theme is applied by an inline script in the layout
// so there is no flash of the wrong theme; this wires up the button and persistence.
(function () {
  var root = document.documentElement;
  var toggle = document.getElementById('themeToggle');
  var storageKey = 'horseRacingTheme';

  function getTheme() {
    return root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    if (toggle) {
      toggle.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
    }
    var meta = document.getElementById('themeColorMeta');
    if (meta) {
      meta.setAttribute('content', theme === 'dark' ? '#0f1216' : '#ffffff');
    }
  }

  applyTheme(getTheme());

  if (!toggle) {
    return;
  }

  toggle.addEventListener('click', function () {
    var next = getTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try {
      localStorage.setItem(storageKey, next);
    } catch (e) {
      /* localStorage unavailable - the choice just won't persist */
    }
  });
})();
