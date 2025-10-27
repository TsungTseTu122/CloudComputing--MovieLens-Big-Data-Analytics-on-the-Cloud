const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function card(node, movie) {
  const tpl = document.getElementById('card-template');
  const el = tpl.content.firstElementChild.cloneNode(true);
  el.querySelector('.title').textContent = movie.title ?? movie.movieId;
  const sub = [];
  if (movie.genres) sub.push(movie.genres);
  if (movie.year) sub.push(String(movie.year));
  if (movie.score != null) sub.push(`★ ${movie.score.toFixed(2)}`);
  el.querySelector('.sub').textContent = sub.join(' • ');
  node.appendChild(el);
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

async function loadPopular() {
  const row = $('#popular-row');
  row.innerHTML = '';
  try {
    const data = await fetchJSON('/popular?topN=50');
    data.forEach((m) => card(row, m));
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
    const genres = await fetchJSON('/genres');
    const top = genres.slice(0, 10);
    top.forEach((g, i) => {
      const b = document.createElement('button');
      b.className = 'chip' + (i === 0 ? ' active' : '');
      b.textContent = g;
      b.onclick = () => {
        $$('.chip').forEach((c) => c.classList.remove('active'));
        b.classList.add('active');
        renderGenre(g);
      };
      list.appendChild(b);
    });
    if (top.length) renderGenre(top[0]);
  } catch (e) {
    list.textContent = 'No genres available';
  }
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
  } catch (e) {
    row.textContent = 'Failed to load';
  }
}

async function handleUserForm() {
  const form = $('#user-form');
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const userId = $('#userId').value.trim();
    const topN = Number($('#topN').value || 20);
    const genres = $('#genres').value.trim();
    const yearFrom = $('#yearFrom').value.trim();
    const yearTo = $('#yearTo').value.trim();
    const row = $('#user-row');
    row.innerHTML = '';
    if (!userId) {
      row.textContent = 'Enter a userId to see personalized recommendations';
      return;
    }
    const params = new URLSearchParams({ topN: String(topN) });
    if (genres) params.set('genres', genres);
    if (yearFrom) params.set('year_from', yearFrom);
    if (yearTo) params.set('year_to', yearTo);
    try {
      const data = await fetchJSON(`/recommendations/user/${encodeURIComponent(userId)}?${params}`);
      if (!data.length) {
        row.textContent = 'No recommendations found for this user.';
      } else {
        data.forEach((m) => card(row, m));
      }
    } catch (e) {
      row.textContent = 'Failed to load recommendations';
    }
  });
}

window.addEventListener('DOMContentLoaded', async () => {
  handleUserForm();
  await loadPopular();
  await loadGenres();
});

