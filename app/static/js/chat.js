// Northwind Policy Assistant chat client.
// Vanilla JS, no framework — keeps the bundle tiny and the demo fast.

(function () {
  'use strict';

  const composer = document.getElementById('composer');
  const input = document.getElementById('question');
  const submitBtn = document.getElementById('submit-btn');
  const conversation = document.getElementById('conversation');
  const intro = document.getElementById('intro');
  const chips = document.querySelectorAll('.chip');

  // Wire up the suggested-question chips to fill the input and submit
  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      input.value = chip.dataset.question;
      composer.requestSubmit();
    });
  });

  composer.addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    // Hide the intro on first question
    if (intro) intro.style.display = 'none';

    appendQuestion(question);
    const thinkingNode = appendThinking();
    setBusy(true);

    const startedAt = performance.now();
    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });
      const data = await response.json();

      thinkingNode.remove();
      appendAnswer(data, performance.now() - startedAt);
    } catch (err) {
      thinkingNode.remove();
      appendAnswer(
        {
          answer: "Sorry, something went wrong reaching the server. Please try again.",
          citations: [],
          refused: true,
          error: String(err),
          latency_ms: performance.now() - startedAt,
        },
        performance.now() - startedAt
      );
    } finally {
      setBusy(false);
      input.value = '';
      input.focus();
    }
  });

  // ---------- Rendering ----------

  function appendQuestion(text) {
    const turn = el('div', { className: 'turn turn-question' });
    const bubble = el('div', { className: 'bubble', textContent: text });
    turn.appendChild(bubble);
    conversation.appendChild(turn);
    scrollToBottom();
  }

  function appendThinking() {
    const node = el('div', { className: 'turn turn-answer' });
    node.innerHTML = `
      <div class="thinking">
        Searching the policy library
        <span class="thinking-dots"><span></span><span></span><span></span></span>
      </div>
    `;
    conversation.appendChild(node);
    scrollToBottom();
    return node;
  }

  function appendAnswer(data, latencyClientMs) {
    const turn = el('div', { className: 'turn turn-answer' });
    if (data.refused) turn.classList.add('refused');

    const body = el('div', { className: 'turn-answer-body' });
    body.innerHTML = formatAnswer(data.answer || '');
    turn.appendChild(body);

    if (data.citations && data.citations.length > 0) {
      turn.appendChild(renderCitations(data.citations));
    }

    const meta = el('div', { className: 'turn-meta' });
    const latencyServer = typeof data.latency_ms === 'number' ? data.latency_ms : 0;
    meta.innerHTML = `
      <span>${data.citations ? data.citations.length : 0} source${(data.citations || []).length === 1 ? '' : 's'}</span>
      <span class="latency">${latencyServer.toFixed(0)} ms</span>
    `;
    turn.appendChild(meta);

    conversation.appendChild(turn);
    scrollToBottom();
  }

  function renderCitations(citations) {
    const list = el('div', { className: 'citations' });
    citations.forEach((c) => {
      const item = el('div', { className: 'citation' });

      const header = el('div', { className: 'citation-header' });
      header.appendChild(el('span', { className: 'citation-id', textContent: c.doc_id }));
      header.appendChild(el('span', { className: 'citation-title', textContent: c.doc_title || '' }));
      if (c.section) {
        header.appendChild(el('span', { className: 'citation-section', textContent: '§ ' + c.section }));
      }
      item.appendChild(header);

      item.appendChild(el('p', { className: 'citation-snippet', textContent: c.snippet || '' }));
      list.appendChild(item);
    });
    return list;
  }

  // ---------- Helpers ----------

  // Highlight inline [POL-XX-NNN] citations in the answer body
  function formatAnswer(text) {
    // Escape HTML first so we can safely wrap with spans
    const escaped = escapeHtml(text);
    return escaped.replace(/\[(POL-[A-Z]{2,4}-\d{3})\]/g, '<span class="cite-tag">$1</span>');
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function setBusy(busy) {
    submitBtn.disabled = busy;
    input.disabled = busy;
  }

  function scrollToBottom() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  }

  function el(tag, props = {}) {
    const node = document.createElement(tag);
    Object.assign(node, props);
    return node;
  }
})();
