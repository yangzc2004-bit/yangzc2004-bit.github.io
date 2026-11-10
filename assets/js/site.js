(function () {
  var root = document.documentElement;
  var header = document.querySelector('.site-header');
  var themeButton = document.querySelector('.theme-toggle');
  var themeMeta = document.querySelector('meta[name="theme-color"]');

  function updateThemeColor() {
    if (themeMeta) themeMeta.content = root.dataset.theme === 'dark' ? '#1d1c19' : '#f5f2eb';
  }

  function updateGiscusTheme() {
    var frame = document.querySelector('iframe.giscus-frame');
    if (!frame) return;
    frame.contentWindow.postMessage({
      giscus: { setConfig: { theme: root.dataset.theme === 'dark' ? 'dark' : 'light' } }
    }, 'https://giscus.app');
  }

  updateThemeColor();
  if (themeButton) {
    themeButton.addEventListener('click', function () {
      var next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      localStorage.setItem('theme', next);
      updateThemeColor();
      updateGiscusTheme();
    });
  }

  function updateHeader() {
    if (header) header.classList.toggle('scrolled', window.scrollY > 12);
  }
  updateHeader();
  window.addEventListener('scroll', updateHeader, { passive: true });

  var dialog = document.getElementById('search-dialog');
  var input = document.getElementById('search-input');
  var results = document.getElementById('search-results');
  var posts = [];
  var loaded = false;
  var selected = -1;

  function setMessage(message, className) {
    results.replaceChildren();
    var paragraph = document.createElement('p');
    paragraph.className = className || 'search-hint';
    paragraph.textContent = message;
    results.appendChild(paragraph);
  }

  function loadPosts() {
    if (loaded) return;
    loaded = true;
    var endpoint = document.getElementById('search-data-url').textContent.trim();
    fetch(endpoint)
      .then(function (response) {
        if (!response.ok) throw new Error('Search data unavailable');
        return response.json();
      })
      .then(function (data) { posts = data; })
      .catch(function () { setMessage('暂时无法搜索，请稍后再试。', 'search-no-results'); });
  }

  function openSearch() {
    if (!dialog || dialog.open) return;
    dialog.showModal();
    input.value = '';
    selected = -1;
    setMessage(posts.length ? '输入关键词，找到一篇笔记。' : '文章发布后，就可以在这里搜索。');
    loadPosts();
    window.setTimeout(function () { input.focus(); }, 30);
  }

  document.querySelectorAll('.search-trigger').forEach(function (button) {
    button.addEventListener('click', openSearch);
  });
  if (dialog) {
    dialog.querySelector('.search-close').addEventListener('click', function () { dialog.close(); });
    dialog.addEventListener('click', function (event) {
      var rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
    });
  }

  document.addEventListener('keydown', function (event) {
    var key = event.key.toLowerCase();
    if ((event.metaKey || event.ctrlKey) && key === 'k') {
      event.preventDefault();
      if (dialog.open) dialog.close(); else openSearch();
    } else if (key === 's' && !event.metaKey && !event.ctrlKey && !event.altKey && !/INPUT|TEXTAREA|SELECT/.test(event.target.tagName) && !event.target.isContentEditable) {
      event.preventDefault();
      openSearch();
    }
  });

  function visibleResults() {
    return Array.from(results.querySelectorAll('.search-result'));
  }

  function selectResult(index) {
    var items = visibleResults();
    if (!items.length) return;
    selected = (index + items.length) % items.length;
    items.forEach(function (item, i) { item.classList.toggle('selected', i === selected); });
    items[selected].scrollIntoView({ block: 'nearest' });
  }

  function renderResults(query) {
    var keyword = query.trim().toLocaleLowerCase();
    selected = -1;
    if (!keyword) {
      setMessage(posts.length ? '输入关键词，找到一篇笔记。' : '文章发布后，就可以在这里搜索。');
      return;
    }
    var matches = posts.filter(function (post) {
      var haystack = [post.title, post.description, (post.tags || []).join(' ')].join(' ').toLocaleLowerCase();
      return haystack.includes(keyword);
    });
    if (!matches.length) {
      setMessage('没有找到相关笔记，换个关键词试试。', 'search-no-results');
      return;
    }
    results.replaceChildren();
    matches.forEach(function (post) {
      var link = document.createElement('a');
      var meta = document.createElement('span');
      var title = document.createElement('strong');
      var description = document.createElement('p');
      link.className = 'search-result';
      link.href = post.url;
      meta.className = 'search-result-meta';
      meta.textContent = post.date + (post.tags && post.tags.length ? '  /  ' + post.tags[0] : '');
      title.textContent = post.title;
      description.textContent = post.description;
      link.append(meta, title, description);
      results.appendChild(link);
    });
  }

  if (input) {
    input.addEventListener('input', function () { renderResults(input.value); });
    input.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowDown') { event.preventDefault(); selectResult(selected + 1); }
      if (event.key === 'ArrowUp') { event.preventDefault(); selectResult(selected - 1); }
      if (event.key === 'Enter') {
        var items = visibleResults();
        if (items.length) { event.preventDefault(); window.location.href = items[Math.max(0, selected)].href; }
      }
    });
  }

  var content = document.getElementById('post-content');
  var tocNav = document.getElementById('table-of-contents');
  if (content && tocNav) {
    var headings = Array.from(content.querySelectorAll('h2, h3'));
    var toc = document.querySelector('.toc');
    if (!headings.length) {
      toc.hidden = true;
      var tocLayout = toc.closest('.post-layout');
      if (tocLayout) tocLayout.classList.add('no-toc');
    } else {
      headings.forEach(function (heading, index) {
        if (!heading.id) heading.id = 'section-' + (index + 1);
        var link = document.createElement('a');
        link.href = '#' + heading.id;
        link.textContent = heading.textContent;
        link.className = heading.tagName === 'H3' ? 'toc-h3' : 'toc-h2';
        tocNav.appendChild(link);
      });
      var tocLinks = Array.from(tocNav.querySelectorAll('a'));
      function updateToc() {
        var active = 0;
        headings.forEach(function (heading, index) {
          if (heading.getBoundingClientRect().top < 160) active = index;
        });
        tocLinks.forEach(function (link, index) { link.classList.toggle('active', index === active); });
      }
      updateToc();
      window.addEventListener('scroll', updateToc, { passive: true });
    }
  }

  var filterButtons = document.querySelectorAll('.category-filter');
  var homeCollection = document.querySelector('.journal .article-collection');
  var filterEmpty = document.querySelector('.filter-empty');
  filterButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      var category = button.dataset.filter;
      filterButtons.forEach(function (item) {
        var active = item === button;
        item.classList.toggle('active', active);
        item.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
      if (!homeCollection) return;
      var visible = 0;
      homeCollection.classList.remove('has-focus');
      homeCollection.querySelectorAll('.post-item').forEach(function (item) { item.classList.remove('is-focused'); });
      homeCollection.querySelectorAll('.category-panel').forEach(function (panel) {
        var show = panel.dataset.category === category;
        panel.hidden = !show;
        if (show) visible += 1;
      });
      if (filterEmpty) filterEmpty.hidden = visible !== 0;
    });
  });

  if (window.matchMedia('(hover: hover)').matches) {
    document.querySelectorAll('.article-collection').forEach(function (collection) {
      var items = collection.querySelectorAll('.post-item');
      function clearFocus() {
        collection.classList.remove('has-focus');
        items.forEach(function (item) { item.classList.remove('is-focused'); });
      }
      items.forEach(function (item) {
        item.addEventListener('mouseenter', function () {
          collection.classList.add('has-focus');
          items.forEach(function (other) { other.classList.toggle('is-focused', other === item); });
        });
        item.addEventListener('mouseleave', function (event) {
          if (!collection.contains(event.relatedTarget)) clearFocus();
        });
      });
      collection.addEventListener('mouseleave', clearFocus);
    });
  }
})();
