const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function card(node, movie) {
  const tpl = document.getElementById('card-template');
  const el = tpl.content.firstElementChild.cloneNode(true);
  if (movie.movieId) el.dataset.mid = String(movie.movieId);
  el.querySelector('.title').textContent = movie.title ?? movie.movieId;
  const sub = [];
  if (movie.genres) sub.push(movie.genres);
  if (movie.year) sub.push(String(movie.year));
  if (movie.score != null) sub.push(`★ ${movie.score.toFixed(2)}`);
  el.querySelector('.sub').textContent = sub.join(' • ');
  const badge = el.querySelector('.badge');
  if (movie.score != null) {
    badge.textContent = `★ ${Number(movie.score).toFixed(2)}`;
    badge.hidden = false;
  }
  if (movie.poster) {
    const p = el.querySelector('.poster');
    p.classList.add('has-img');
    p.style.backgroundImage = `url('${movie.poster}')`;
    p.textContent = '';
  }
  el.addEventListener('mouseenter', (ev) => showTooltip(ev, movie));
  el.addEventListener('mousemove', (ev) => showTooltip(ev, movie));
  el.addEventListener('mouseleave', hideTooltip);
  // Click feedback disabled to keep focus on list (wishlist) only
  // favorites toggle
  const favBtn = el.querySelector('.fav');
  const favs = getFavorites();
  if (favs.has(movie.movieId)) favBtn.classList.add('active'), favBtn.textContent = '❤';
  favBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const set = getFavorites();
    if (set.has(movie.movieId)) {
      set.delete(movie.movieId);
      favBtn.classList.remove('active');
      favBtn.textContent = '♡';
    } else {
      set.add(movie.movieId);
      favBtn.classList.add('active');
      favBtn.textContent = '❤';
      sendFeedback('list', movie.movieId);
    }
    saveFavorites(set);
    renderMyList();
  });
  node.appendChild(el);
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

async function loadPopular() {
  const row = $('#popular-row');
  showSkeleton(row, 8);
  try {
    const data = await fetchJSON('/popular?topN=50');
    row.innerHTML = '';
    data.forEach((m) => card(row, m));
    const posters = await postersFor(data);
    data.forEach(m => {
      const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
      if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
    });
  } catch (e) {
    row.textContent = 'Failed to load popular titles';
  }
}

async function loadGenres() {
  const list = $('#genres-list');
  list.innerHTML = '';
  const rows = $('#genre-rows');
  rows.innerHTML = '';
  try {
    const gsrc = await fetchJSON('/genres');
    let genres = Array.isArray(gsrc) ? gsrc.slice() : [];
    // Deduplicate (case-insensitive) and move '(no genre listed)' to the bottom
    const seen = new Set();
    genres = genres.filter(g => { const k = g.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });
    const noneIdx = genres.findIndex(g => g.toLowerCase() === '(no genre listed)');
    if (noneIdx >= 0) { const [t] = genres.splice(noneIdx, 1); genres.push(t); }
    const top = genres.slice(0, 10);
    top.forEach((g, i) => {
      const b = document.createElement('button');
      b.className = 'chip' + (i === 0 ? ' active' : '');
      b.textContent = g;
      b.onclick = () => { window.location.href = `/ui/genre.html?g=${encodeURIComponent(g)}`; };
      list.appendChild(b);
    });
    // Do not auto-render a horizontal row; use dedicated genre pages
    // Populate sidebar genres
    const side = document.getElementById('side-genres');
    if (side) {
      side.innerHTML = '';
      genres.forEach((g) => {
        const a = document.createElement('a');
        a.href = `/ui/genre.html?g=${encodeURIComponent(g)}`;
        a.className = 'side-link';
        a.textContent = g;
        side.appendChild(a);
      });
      const title = document.getElementById('side-genres-title');
      if (title) {
        title.style.cursor = 'pointer';
        title.onclick = () => {
          side.classList.toggle('collapsed');
          title.textContent = side.classList.contains('collapsed') ? 'Genres ▸' : 'Genres ▾';
        };
      }
    }
  } catch (e) {
    list.textContent = 'No genres available';
  }
}

async function initUsersSelect() {
  const btn = document.getElementById('userBtn');
  const menu = document.getElementById('userMenu');
  const search = document.getElementById('userSearch');
  const list = document.getElementById('userListBox');
  if (!btn || !menu || !search || !list) return;
  try {
    let users = await fetchJSON('/users?limit=500');
    // Defensive sort: numeric ascending when possible
    users = (Array.isArray(users) ? users : []).slice().sort((a,b) => {
      const sa=String(a).trim(), sb=String(b).trim();
      const da=/^\d+$/.test(sa), db=/^\d+$/.test(sb);
      if (da && db) return Number(sa) - Number(sb);
      if (da && !db) return -1;
      if (!da && db) return 1;
      return sa.localeCompare(sb);
    });
    // Build list of items
    function render(q = '') {
      const query = String(q).trim().toLowerCase();
      list.innerHTML = '';
      const frag = document.createDocumentFragment();
      const data = users.filter(u => !query || String(u).toLowerCase().includes(query));
      data.forEach(u => {
        const d = document.createElement('button');
        d.type = 'button'; d.className = 'dropdown-item'; d.textContent = u; d.setAttribute('role','option');
        d.onclick = () => { btn.dataset.user = String(u); btn.textContent = `${u} ▾`; menu.hidden = true; localStorage.setItem('ml_user', String(u)); };
        frag.appendChild(d);
      });
      if (!data.length) {
        const empty = document.createElement('div'); empty.className='empty-center'; empty.textContent='No users'; frag.appendChild(empty);
      }
      list.appendChild(frag);
    }
    render('');
    // Open/close behavior
    btn.addEventListener('click', () => {
      menu.hidden = !menu.hidden;
      btn.classList.toggle('open', !menu.hidden);
      btn.setAttribute('aria-expanded', String(!menu.hidden));
      if (!menu.hidden) { search.focus(); }
    });
    document.addEventListener('click', (e) => {
      if (!menu.contains(e.target) && e.target !== btn) {
        menu.hidden = true;
        btn.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
      }
    });
    search.addEventListener('input', () => render(search.value));
    // Restore last user
    const last = localStorage.getItem('ml_user'); if (last) { btn.dataset.user = last; btn.textContent = `${last} ▾`; }
  } catch {}
}

async function renderGenre(genre) {
  const rows = $('#genre-rows');
  rows.innerHTML = '';
  const section = document.createElement('div');
  const h = document.createElement('h3');
  h.textContent = genre;
  section.appendChild(h);
  const row = document.createElement('div');
  row.className = 'row';
  section.appendChild(row);
  rows.appendChild(section);
  try {
    const data = await fetchJSON(`/popular?topN=20&genres=${encodeURIComponent(genre)}`);
    data.forEach((m) => card(row, m));
    const posters = await postersFor(data);
    data.forEach(m => {
      const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
      if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
    });
  } catch (e) {
    row.textContent = 'Failed to load';
  }
}

async function handleUserForm() {
  const form = $('#user-form');
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const userBtn = document.getElementById('userBtn');
    const userId = userBtn ? String((userBtn.dataset.user||'').trim()) : '';
    const topN = 20;
    const genres = $('#genres').value.trim();
    const yearFromEl = document.getElementById('yearFrom');
    const yearToEl = document.getElementById('yearTo');
    const yearFrom = yearFromEl ? String((yearFromEl.value||'').trim()) : '';
    const yearTo = yearToEl ? String((yearToEl.value||'').trim()) : '';
    const row = $('#user-row');
    localStorage.setItem('ml_user', userId || '');
    showSkeleton(row, 8);
    const params = new URLSearchParams({ topN: String(topN) });
    if (genres) params.set('genres', genres);
    if (yearFrom) params.set('year_from', yearFrom);
    if (yearTo) params.set('year_to', yearTo);
    try {
      const data = userId
        ? await fetchJSON(`/recommendations/user/${encodeURIComponent(userId)}?${params}`)
        : await fetchJSON(`/popular?${params}`);
      if (!data.length) {
        row.textContent = 'No recommendations found for this user.';
      } else {
        row.innerHTML = '';
        data.forEach((m) => card(row, m));
        const posters = await postersFor(data);
        data.forEach(m => {
          const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
          if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
        });
      }
      // no continue-watching UI
    } catch (e) {
      row.textContent = 'Failed to load recommendations';
    }
  });
}

async function postersFor(items) {
  try {
    const ids = items.slice(0, 12).map((m) => m.movieId).filter(Boolean);
    if (!ids.length) return {};
    return await fetchJSON(`/posters?movieIds=${encodeURIComponent(ids.join(','))}`);
  } catch {
    return {};
  }
}

// Populate year range selects from API /years
async function initYearSelectors() {
  const yfrom = document.getElementById('yearFrom');
  const yto = document.getElementById('yearTo');
  if (!yfrom || !yto) return;
  try {
    const data = await fetchJSON('/years');
    const years = Array.isArray(data.years) ? data.years : [];
    function fill() {
      yfrom.innerHTML=''; yto.innerHTML='';
      const optAllFrom = document.createElement('option'); optAllFrom.value=''; optAllFrom.textContent='From'; yfrom.appendChild(optAllFrom);
      const optAllTo = document.createElement('option'); optAllTo.value=''; optAllTo.textContent='To'; yto.appendChild(optAllTo);
      years.forEach(y=>{ const o=document.createElement('option'); o.value=String(y); o.textContent=String(y); yfrom.appendChild(o.cloneNode(true)); yto.appendChild(o); });
    }
    fill();
    function clampTo() {
      const fv = parseInt(yfrom.value||'0',10);
      Array.from(yto.options).forEach((o,i)=>{ if (i===0) return; o.disabled = !!yfrom.value && parseInt(o.value,10) < fv; });
    }
    function clampFrom() {
      const tv = parseInt(yto.value||'0',10);
      Array.from(yfrom.options).forEach((o,i)=>{ if (i===0) return; o.disabled = !!yto.value && parseInt(o.value,10) > tv; });
    }
    yfrom.addEventListener('change', clampTo);
    yto.addEventListener('change', clampFrom);
  } catch {}
}

// Build genre chips and year preset chips under the Recommendations panel
async function initUserFilters() {
  // Genre chips populate the hidden text input #genres for compatibility
  const box = document.getElementById('user-genres');
  if (box) {
    try {
      const gsrc = await fetchJSON('/genres');
      let genres = Array.isArray(gsrc) ? gsrc.slice() : [];
      const seen2 = new Set();
      genres = genres.filter(g => { const k = g.toLowerCase(); if (seen2.has(k)) return false; seen2.add(k); return true; });
      const idx = genres.findIndex(g => g.toLowerCase() === '(no genre listed)');
      if (idx >= 0) { const [t] = genres.splice(idx, 1); genres.push(t); }
      const picked = new Set();
      genres.slice(0, 20).forEach((g) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip';
        b.textContent = g;
        b.onclick = () => {
          if (picked.has(g)) { picked.delete(g); b.classList.remove('active'); }
          else { picked.add(g); b.classList.add('active'); }
          const inp = document.getElementById('genres');
          if (inp) inp.value = Array.from(picked).join(', ');
        };
        box.appendChild(b);
      });
    } catch {
      // ignore
    }
  }

  // Year preset chips write to #yearFrom and #yearTo inputs
  const yp = document.getElementById('year-presets');
  if (yp) {
    const presets = [
      {label: '1980s', from: 1980, to: 1989},
      {label: '1990s', from: 1990, to: 1999},
      {label: '2000s', from: 2000, to: 2009},
      {label: '2010s', from: 2010, to: 2019},
      {label: '2020s', from: 2020, to: 2029},
      {label: 'All', from: '', to: ''},
    ];
    presets.forEach((p) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.textContent = p.label;
      b.onclick = () => {
        const yf = document.getElementById('yearFrom');
        const yt = document.getElementById('yearTo');
        if (yf) yf.value = p.from;
        if (yt) yt.value = p.to;
      };
      yp.appendChild(b);
    });
  }
}

function handleBrowse() {
  const form = $('#browse-form');
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const q = $('#q').value.trim();
    const topN = 50;
    const row = $('#browse-row');
    showSkeleton(row, 8);
    try {
      const data = await fetchJSON(`/movies?topN=${topN}${q ? `&q=${encodeURIComponent(q)}` : ''}`);
      row.innerHTML = '';
      if (!data.length) { row.textContent = 'No results.'; return; }
      data.forEach((m) => card(row, m));
      const posters = await postersFor(data);
      data.forEach(m => {
        const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
        if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
      });
    } catch (e) {
      row.textContent = 'Search failed';
    }
  });
  // Enter to search
  $('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); form.requestSubmit(); } });
}

window.addEventListener('DOMContentLoaded', async () => {
  await initUsersSelect();
  handleUserForm();
  initUserFilters();
  initYearSelectors();
  handleBrowse();
  await loadPopular();
  await loadMostClicked();
  await loadGenres();
  renderMyList();
  enableKeyboardScroll();
  // Restore last user
  // handled in initUsersSelect
  // View mode toggle: posters/list
  const savedMode = localStorage.getItem('ml_mode');
  if (savedMode === 'list') { document.body.classList.add('list-mode'); }
  const modeBtn = document.getElementById('mode-toggle');
  const syncIcon = () => { modeBtn.textContent = document.body.classList.contains('list-mode') ? '📄' : '🖼️'; };
  syncIcon();
  modeBtn.addEventListener('click', () => {
    document.body.classList.toggle('list-mode');
    const isList = document.body.classList.contains('list-mode');
    localStorage.setItem('ml_mode', isList ? 'list' : 'posters');
    syncIcon();
  });
  // TopN control
  initTopN();
});

// Favorites (localStorage)
const FAV_KEY = 'ml_favorites';
function getFavorites() {
  try { return new Set(JSON.parse(localStorage.getItem(FAV_KEY) || '[]')); } catch { return new Set(); }
}
function saveFavorites(set) {
  localStorage.setItem(FAV_KEY, JSON.stringify(Array.from(set)));
}
async function renderMyList() {
  const row = document.getElementById('mylist-row');
  if (!row) return;
  row.innerHTML = '';
  const ids = Array.from(getFavorites());
  if (!ids.length) { const d = document.createElement('div'); d.className='empty-center'; d.textContent='Add titles with the heart to build your list.'; row.appendChild(d); return; }
  try {
    const items = await fetchJSON(`/movies/by_ids?movieIds=${encodeURIComponent(ids.join(','))}`);
    const posters = await postersFor(items);
    items.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch { row.textContent = 'Failed to load list'; }
}

// Keyboard row scrolling
function enableKeyboardScroll() {
  const rows = $$('.row');
  rows.forEach((r) => r.addEventListener('mouseenter', () => rows.forEach((x) => x.classList.toggle('active', x===r))));
  window.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const active = document.querySelector('.row.active');
    if (!active) return;
    const dx = e.key === 'ArrowRight' ? 300 : -300;
    active.scrollBy({ left: dx, behavior: 'smooth' });
  });
}

// TopN chips + arrows
function initTopN() {
  const chipsBox = document.getElementById('topn-chips');
  const left = document.getElementById('topn-left');
  const right = document.getElementById('topn-right');
  const hidden = document.getElementById('topN');
  if (!chipsBox || !left || !right || !hidden) return;
  const values = [5,10,15,20];
  let idx = Math.max(0, values.indexOf(Number(hidden.value) || 20));
  function render() {
    chipsBox.innerHTML='';
    values.forEach((v,i)=>{
      const b=document.createElement('button'); b.type='button'; b.className='chip'+(i===idx?' active':''); b.textContent=String(v);
      b.onclick=()=>{ idx=i; hidden.value=String(values[idx]); render(); };
      chipsBox.appendChild(b);
    });
  }
  left.onclick = () => { idx = Math.max(0, idx-1); hidden.value=String(values[idx]); render(); };
  right.onclick = () => { idx = Math.min(values.length-1, idx+1); hidden.value=String(values[idx]); render(); };
  render();
}

// Continue Watching
async function renderContinue(userId) {
  const row = document.getElementById('continue-row');
  if (!row) return;
  showSkeleton(row, 6);
  try {
    const evs = await fetchJSON(`/history?userId=${encodeURIComponent(userId)}&topN=50`);
    const ids = Array.from(new Set(evs.map((e) => e.movieId))).slice(0, 20);
    if (!ids.length) { row.textContent = 'No recent activity.'; return; }
    const items = await fetchJSON(`/movies/by_ids?movieIds=${encodeURIComponent(ids.join(','))}`);
    const posters = await postersFor(items);
    row.innerHTML='';
    items.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch { row.textContent = 'No recent activity.'; }
}

// Skeleton helpers
function showSkeleton(row, n) {
  row.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const d = document.createElement('div');
    d.className = 'card skeleton';
    const p = document.createElement('div');
    p.className = 'poster';
    const t = document.createElement('div');
    t.className = 'meta title';
    d.appendChild(p); d.appendChild(t);
    row.appendChild(d);
  }
}

// Most Clicked (global) and per-user
async function loadMostClicked() {
  const row = document.getElementById('clicked-row');
  if (!row) return;
  showSkeleton(row, 8);
  try {
    const items = await fetchJSON('/feedback/summary?topN=20&window_days=30&half_life_days=14&w_click=1&w_list=3&blend_baseline=0.2');
    const posters = await postersFor(items);
    row.innerHTML='';
    items.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch { row.textContent = 'No click data yet.'; }
}

async function loadMostClickedForUser(userId) {
  const cont = document.getElementById('continue-row'); // ensure continued loaded separately
  try {
    const row = document.getElementById('user-clicked-row');
    if (!row) return;
    showSkeleton(row, 8);
    const items = await fetchJSON(`/feedback/summary?userId=${encodeURIComponent(userId)}&topN=20`);
    const posters = await postersFor(items);
    row.innerHTML='';
    items.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch {}
}

// Tooltip
const tooltip = document.getElementById('tooltip');
function showTooltip(ev, movie) {
  if (!tooltip) return;
  $('.tt-title', tooltip).textContent = movie.title ?? movie.movieId;
  const bits = [];
  if (movie.year) bits.push(movie.year);
  if (movie.genres) bits.push(movie.genres);
  if (movie.score != null) bits.push(`score ${Number(movie.score).toFixed(2)}`);
  $('.tt-sub', tooltip).textContent = bits.join(' • ');
  tooltip.hidden = false;
  const pad = 12;
  const x = Math.min(window.innerWidth - tooltip.offsetWidth - pad, ev.clientX + pad);
  const y = Math.min(window.innerHeight - tooltip.offsetHeight - pad, ev.clientY + pad);
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}
function hideTooltip() { if (tooltip) tooltip.hidden = true; }

// Feedback helper
async function sendFeedback(action, movieId) {
  try {
    const uid = localStorage.getItem('ml_user') || 'guest';
    await fetch('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ userId: uid, movieId, action }) });
  } catch {}
}
