const $ = (s, r=document)=>r.querySelector(s);

function card(node, movie){
  const tpl = document.getElementById('card-template');
  const el = tpl.content.firstElementChild.cloneNode(true);
  el.querySelector('.title').textContent = movie.title ?? movie.movieId;
  const sub = [];
  if (movie.genres) sub.push(movie.genres);
  if (movie.year) sub.push(String(movie.year));
  if (movie.score != null) sub.push(`★ ${Number(movie.score).toFixed(2)}`);
  el.querySelector('.sub').textContent = sub.join(' • ');
  if (movie.poster) {
    const p = el.querySelector('.poster');
    p.classList.add('has-img');
    p.style.backgroundImage = `url('${movie.poster}')`;
    p.textContent='';
  }
  node.appendChild(el);
}

async function fetchJSON(url){ const r=await fetch(url); if(!r.ok) throw new Error(r.status); return r.json(); }

function getQuery(){ const p=new URLSearchParams(location.search); return { g: p.get('g')||'' } }

function showSkeleton(row, n){ row.innerHTML=''; for(let i=0;i<n;i++){ const d=document.createElement('div'); d.className='card skeleton'; const p=document.createElement('div'); p.className='poster'; const t=document.createElement('div'); t.className='meta title'; d.appendChild(p); d.appendChild(t); row.appendChild(d); } }

async function loadGenre(){
  const { g } = getQuery();
  const title = $('#genre-title');
  title.textContent = g || 'Genre';
  const row = $('#genre-row');
  showSkeleton(row, 8);
  try{
    const token = (g && g.toLowerCase()==='(genres not listed)') ? 'No genre listed' : g;
    const items = await fetchJSON(`/movies?topN=200&genres=${encodeURIComponent(token)}`);
    const ids = items.map(m=>m.movieId).filter(Boolean);
    let posters = {};
    try{ posters = await fetchJSON(`/posters?movieIds=${encodeURIComponent(ids.join(','))}`);}catch{}
    row.innerHTML='';
    items.map(m=>({...m, poster: posters[m.movieId]})).forEach(m=>card(row,m));
  }catch{ row.textContent='Failed to load'; }
}

window.addEventListener('DOMContentLoaded', loadGenre);

