const $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));

function statusClass(status){
  if(['PRINTING','ONLINE','DONE','IDLE'].includes(status))return'ok';
  if(['PAUSED','UNKNOWN'].includes(status))return'warn';
  if(['ERROR','OFFLINE'].includes(status))return'bad';
  return'muted';
}

function fmt(ts){
  if(!ts)return'—';
  const d=new Date(ts);
  if(Number.isNaN(d.getTime()))return'—';
  return d.toLocaleString('ru-RU',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

function deviceCard(d){
  const data=d.data||{};
  return `<article class="device-card">
    <div class="device-top">
      <div><h3>${esc(d.device_id||'Устройство')}</h3><div class="device-id">${esc(d.last_event||'нет событий')}</div></div>
      <span class="pill ${statusClass(d.status)}">${esc(d.status||'UNKNOWN')}</span>
    </div>
    <div class="device-message">${esc(d.message||'Нет сообщения')}</div>
    <div class="device-meta">
      <div><span>Последний сигнал</span><b>${esc(fmt(d.last_seen))}</b></div>
      <div><span>Серьёзность</span><b>${esc(d.severity||'info')}</b></div>
      <div><span>Движение</span><b>${data.motion_score==null?'—':Number(data.motion_score).toFixed(3)}</b></div>
      <div><span>Яркость</span><b>${data.brightness==null?'—':Number(data.brightness).toFixed(1)}</b></div>
    </div>
  </article>`;
}

function eventRow(e){
  return `<article class="event ${esc(e.severity||'info')}">
    <div class="event-top"><span class="event-type">${esc(e.type)}</span><span class="event-time">${esc(fmt(e.ts))}</span></div>
    <div class="event-message">${esc(e.message||'')}</div>
  </article>`;
}

async function getJson(url,options){
  const r=await fetch(url,options);
  if(!r.ok)throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function load(){
  try{
    const [health,status,events]=await Promise.all([
      getJson('/health'),
      getJson('/api/v1/workshop/status'),
      getJson('/api/v1/events?limit=50')
    ]);
    $('apiState').textContent=`API v${health.version||'?'} · ONLINE`;
    $('apiState').className='pill ok';
    const devices=status.devices||[];
    const alerts=status.alerts||[];
    $('deviceCount').textContent=devices.length;
    $('printingCount').textContent=devices.filter(x=>x.status==='PRINTING').length;
    $('alertCount').textContent=alerts.length;
    $('updatedAt').textContent=new Date().toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit'});
    $('devices').innerHTML=devices.map(deviceCard).join('')||'<div class="empty">Нет данных от оборудования</div>';
    $('events').innerHTML=(events||[]).map(eventRow).join('')||'<div class="empty">Событий пока нет</div>';

    const ad5x=devices.find(x=>String(x.device_id||'').includes('ad5x'))||devices[0];
    if(ad5x){
      $('cameraCaption').textContent=`${ad5x.device_id} · последний сигнал ${fmt(ad5x.last_seen)}`;
      $('cameraState').textContent=ad5x.status||'UNKNOWN';
      $('cameraState').className=`pill ${statusClass(ad5x.status)}`;
    }
  }catch(error){
    $('apiState').textContent='API · OFFLINE';
    $('apiState').className='pill bad';
    $('updatedAt').textContent='ошибка';
    console.error(error);
  }
}

$('refreshBtn').addEventListener('click',load);
$('assistantForm').addEventListener('submit',async e=>{
  e.preventDefault();
  const text=$('assistantInput').value.trim();
  if(!text)return;
  const button=e.currentTarget.querySelector('button');
  button.disabled=true;
  $('assistantAnswer').textContent='Джарвис анализирует состояние цеха…';
  try{
    const data=await getJson('/api/v1/assistant/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    $('assistantAnswer').textContent=data.answer||data.response||JSON.stringify(data,null,2);
    $('assistantInput').value='';
  }catch(error){
    $('assistantAnswer').textContent=`Ошибка запроса: ${error.message}`;
  }finally{button.disabled=false}
});

load();
setInterval(load,5000);
