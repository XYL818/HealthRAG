/* 知健前端：消费 /api/stream 的 SSE，边收边渲染。 */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var form = $('ask-form');
  var input = $('q');
  var askBtn = $('ask-btn');
  var answerCard = $('answer-card');
  var answerEl = $('answer');
  var timerEl = $('timer');
  var refusedNote = $('refused-note');
  var sourcesCard = $('sources-card');
  var sourcesEl = $('sources');
  var srcCount = $('src-count');
  var traceCard = $('trace-card');
  var traceEl = $('trace');
  var errorEl = $('error');

  var buffer = '';        // 累积的答案文本
  var sources = [];       // 当前来源
  var t0 = 0;
  var tick = null;

  /* ---------------------------------------------------------- 工具 */

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // 把答案文本转成 HTML：[1] 变成可点的引用角标，**粗体** 变 <strong>
  function renderAnswer(text) {
    var html = escapeHtml(text);
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\[(\d+)\]/g, function (_, n) {
      var idx = parseInt(n, 10);
      var has = sources.some(function (s) { return s.n === idx; });
      if (!has) { return '[' + n + ']'; }
      return '<span class="cite" data-n="' + idx + '">' + n + '</span>';
    });
    return html;
  }

  function show(el) { el.classList.remove('hidden'); }
  function hide(el) { el.classList.add('hidden'); }

  function startTimer() {
    t0 = Date.now();
    timerEl.textContent = '0.0s';
    tick = setInterval(function () {
      timerEl.textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's';
    }, 100);
  }

  function stopTimer() {
    if (tick) { clearInterval(tick); tick = null; }
  }

  function showError(msg) {
    errorEl.textContent = '出错了：' + msg;
    show(errorEl);
  }

  /* ---------------------------------------------------------- 渲染 */

  function renderSources(list, refused) {
    sources = list || [];
    sourcesEl.innerHTML = '';
    srcCount.textContent = sources.length ? sources.length + ' 条' : '';

    if (refused || !sources.length) {
      hide(sourcesCard);
      return;
    }

    sources.forEach(function (s) {
      var scoreTxt = (s.score === null || s.score === undefined)
        ? 'RRF ' + (s.rrf_score || 0).toFixed(4)
        : Number(s.score).toFixed(3);

      var head = s.title + (s.section && s.section !== s.title ? ' > ' + s.section : '');

      var el = document.createElement('div');
      el.className = 'src';
      el.id = 'src-' + s.n;
      el.innerHTML =
        '<div class="src-head">' +
          '<span class="src-n">' + s.n + '</span>' +
          '<span class="src-title">' + escapeHtml(s.title) + '</span>' +
          '<span class="src-sec">' + escapeHtml(s.section || '') + '</span>' +
          '<span class="src-score">' + scoreTxt + '</span>' +
        '</div>' +
        '<div class="src-text">' + escapeHtml(s.text) + '</div>' +
        '<span class="src-more">展开全文</span>' +
        ' <a href="' + escapeHtml(s.url || '#') + '" target="_blank" rel="noopener">查看原文</a>';

      el.querySelector('.src-more').addEventListener('click', function () {
        var on = el.classList.toggle('expanded');
        this.textContent = on ? '收起' : '展开全文';
      });

      // 点击角标跳到对应来源
      sourcesEl.appendChild(el);
    });

    show(sourcesCard);
  }

  function renderTrace(list) {
    if (!list || !list.length) { hide(traceCard); return; }
    var rows = list.map(function (s, i) {
      var d = s.dense_rank, b = s.bm25_rank;
      var scoreTxt = (s.score === null || s.score === undefined)
        ? '—' : Number(s.score).toFixed(3);
      var flag = (i === 0) ? ' class="hit"' : '';
      return '<tr>' +
        '<td class="num">' + s.n + '</td>' +
        '<td>' + escapeHtml(s.title) + '</td>' +
        '<td class="num' + (d ? '' : ' miss') + '">' + (d || '—') + '</td>' +
        '<td class="num' + (b ? '' : ' miss') + '">' + (b || '—') + '</td>' +
        '<td class="num"' + flag + '>' + scoreTxt + '</td>' +
        '</tr>';
    }).join('');

    traceEl.innerHTML =
      '<table><thead><tr>' +
        '<th style="width:34px">#</th><th>来源</th>' +
        '<th class="num" style="width:56px">向量名次</th>' +
        '<th class="num" style="width:56px">BM25名次</th>' +
        '<th class="num" style="width:64px">重排分</th>' +
      '</tr></thead><tbody>' + rows + '</tbody></table>' +
      '<p style="color:#8b9a96;font-size:11.5px;margin:8px 0 0">' +
      '「—」表示该片段只被一条通道召回。两路都命中的片段通常更可靠。</p>';
    show(traceCard);
  }

  /* ---------------------------------------------------------- SSE */

  function handleEvent(name, data) {
    if (name === 'sources') {
      renderSources(data.sources, data.refused);
      renderTrace(data.sources);
      if (data.refused) { show(refusedNote); } else { hide(refusedNote); }
    } else if (name === 'token') {
      buffer += data.t;
      answerEl.innerHTML = renderAnswer(buffer) + '<span class="cursor"></span>';
    } else if (name === 'done') {
      answerEl.innerHTML = renderAnswer(buffer);
      stopTimer();
      timerEl.textContent = (data.elapsed || 0).toFixed(1) + 's';
      bindCites();
    } else if (name === 'error') {
      stopTimer();
      showError(data.message);
    }
  }

  function bindCites() {
    answerEl.querySelectorAll('.cite').forEach(function (c) {
      c.addEventListener('click', function () {
        var target = $('src-' + c.dataset.n);
        if (!target) { return; }
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        target.classList.add('flash');
        setTimeout(function () { target.classList.remove('flash'); }, 1200);
      });
    });
  }

  function parseChunk(raw) {
    // 一个 SSE 事件块形如：
    //   event: token
    //   data: {"t":"..."}
    var name = 'message', dataStr = '';
    raw.split('\n').forEach(function (line) {
      if (line.indexOf('event:') === 0) { name = line.slice(6).trim(); }
      else if (line.indexOf('data:') === 0) { dataStr += line.slice(5).trim(); }
    });
    if (!dataStr) { return; }
    try { handleEvent(name, JSON.parse(dataStr)); } catch (e) { /* 忽略半截 JSON */ }
  }

  function ask(question) {
    if (!question || question.length < 2) { return; }

    hide(errorEl);
    hide(answerCard);
    hide(sourcesCard);
    hide(traceCard);
    hide(refusedNote);
    buffer = '';
    sources = [];
    answerEl.innerHTML = '<span class="cursor"></span>';
    show(answerCard);
    startTimer();

    askBtn.disabled = true;
    askBtn.textContent = '生成中';

    fetch('/api/stream?q=' + encodeURIComponent(question))
      .then(function (resp) {
        if (!resp.ok || !resp.body) {
          throw new Error('HTTP ' + resp.status);
        }
        var reader = resp.body.getReader();
        var decoder = new TextDecoder('utf-8');
        var pending = '';

        function pump() {
          return reader.read().then(function (res) {
            if (res.done) { return; }
            pending += decoder.decode(res.value, { stream: true });

            var parts = pending.split('\n\n');
            pending = parts.pop();          // 最后一段可能不完整，留着下轮拼
            parts.forEach(parseChunk);
            return pump();
          });
        }
        return pump();
      })
      .catch(function (e) { stopTimer(); showError(e.message); })
      .finally(function () {
        askBtn.disabled = false;
        askBtn.textContent = '提问';
      });
  }

  /* ---------------------------------------------------------- 事件绑定 */

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    ask(input.value.trim());
  });

  document.querySelectorAll('.chip').forEach(function (c) {
    c.addEventListener('click', function () {
      input.value = c.textContent;
      ask(c.textContent);
    });
  });

  $('trace-toggle').addEventListener('click', function () {
    traceCard.classList.toggle('collapsed');
    this.classList.toggle('collapsed');
  });

  // 首页统计
  fetch('/api/stats')
    .then(function (r) { return r.json(); })
    .then(function (s) {
      if (!s || !s.num_chunks) { return; }
      $('stats').innerHTML =
        '<div><b>' + s.num_chunks.toLocaleString() + '</b>检索片段</div>' +
        '<div><b>' + s.num_docs + '</b>篇文档</div>' +
        '<div><b>' + s.dim + '</b>向量维度</div>' +
        '<div><b>' + s.avg_chars + '</b>平均块长</div>' +
        '<div><b>' + (s.embed_model || '').split('/').pop() + '</b>向量模型</div>';
    })
    .catch(function () { /* 统计失败不影响主流程 */ });
})();
