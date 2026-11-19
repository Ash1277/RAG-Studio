const $ = (id) => document.getElementById(id);
let currentSession = null;
let currentMessages = [];
let busy = false;
let selectedSourceIds = new Set();
let knownDocsById = new Map();
let hasPromptedKeySetup = false;
const views = {
 home: {title:'Home', subtitle:'', eyebrow:''},
 chat: {title:'Ask documents', subtitle:'', eyebrow:''},
 research: {title:'Research & compare', subtitle:'Synthesize evidence across documents and investigate contradictions.', eyebrow:'DOCUMENT RESEARCH'},
 study: {title:'Study mode', subtitle:'Create revision material or test yourself.', eyebrow:'LEARN FROM YOUR SOURCES'},
 history: {title:'Conversation history', subtitle:'Reopen saved chats whenever you need them.', eyebrow:'YOUR WORKSPACE'}
};
async function api(url, options={}) {
 const headers = options.body && !(options.body instanceof FormData) ? {'Content-Type':'application/json',...(options.headers||{})} : (options.headers||{});
 const res = await fetch(url,{...options,headers});
 const type=res.headers.get('content-type')||'';
 const data=type.includes('application/json')?await res.json():await res.text();
 if(!res.ok){const detail=data&&data.detail;const message=typeof detail==='string'?detail:(detail&&detail.message?`${detail.message}\nReason: ${detail.reason||'Not provided'}\n${detail.suggestion||''}`:`Request failed (${res.status})`);throw new Error(message)}
 return data;
}
function toast(message,error=false){const el=document.createElement('div');el.className='toast'+(error?' error':'');el.textContent=message;$('toastRegion').appendChild(el);setTimeout(()=>el.remove(),4800)}
function setBusy(value){busy=value;['sendButton','chooseFiles','addUrl','runResearch','generateStudy','generateQuiz'].forEach(id=>{if($(id))$(id).disabled=value});$('sendButton').textContent=value?'…':'↑'}
function applyTheme(theme, persistBackend=false){
 const allowed=['ocean','forest','lavender','sand','midnight','ghost','ghost-dark','dark'];
 if(!allowed.includes(theme))theme='ghost';
 document.body.dataset.theme=theme;
 localStorage.setItem('rag-v3-theme',theme);
 if($('settingsTheme'))$('settingsTheme').value=theme;
 document.querySelectorAll('[data-theme-pick]').forEach(btn=>btn.classList.toggle('active',btn.dataset.themePick===theme));
 document.querySelectorAll('[data-swiss-mode]').forEach(btn=>btn.classList.toggle('active',btn.dataset.swissMode===theme));
 if(persistBackend){
  api('/api/settings/preferences',{method:'PUT',body:JSON.stringify({theme,default_top_k:5,use_reranker:Boolean($('rerankToggle')?.checked)})}).catch(()=>{});
 }
}
function updateSelectedSourcesBadge(){
 const badge=$('selectedSourcesBadge');
 if(!badge)return;
 const ids=[...selectedSourceIds];
 const count=ids.length;
 if(count===0){
  badge.textContent='＋ Pick a source ⌄';
 }else if(count===1){
  const name=knownDocsById.get(ids[0])||'1 source';
  const shortName=name.length>28?name.slice(0,27)+'…':name;
  badge.textContent=`📄 ${shortName} ⌄`;
 }else{
  badge.textContent=`📄 ${count} sources · Switch ⌄`;
 }
}
function setView(name){
 document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active',v.id===`view-${name}`));
 document.querySelectorAll('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.view===name));
 if($('pageTitle'))$('pageTitle').textContent=views[name].title;
 if($('pageSubtitle'))$('pageSubtitle').textContent=views[name].subtitle;
 if($('eyebrow'))$('eyebrow').textContent=views[name].eyebrow;
 if($('toggleDrawerBtn'))$('toggleDrawerBtn').style.display=name==='chat'?'none':'inline-flex';
 if(name==='history')renderHistory();
 if(['chat','study','research'].includes(name))refreshDocuments();
}
function formatBytes(bytes){if(bytes==null)return 'Web source';if(bytes<1024)return `${bytes} B`;if(bytes<1048576)return `${(bytes/1024).toFixed(1)} KB`;return `${(bytes/1048576).toFixed(1)} MB`}
function makeGroundingBadge(grounding){
 if(!grounding||!grounding.label)return null;
 const badge=document.createElement('div');
 badge.className=`grounding-badge grounding-${grounding.level||'medium'}`;
 const icon=grounding.level==='high'?'✓':(grounding.level==='medium'?'◑':'⚠');
 badge.textContent=`${icon} ${grounding.label}`;
 if(grounding.warning)badge.title=grounding.warning;
 return badge;
}
function makeSourceBox(sources){
 if(!sources||!sources.length)return null;
 const box=document.createElement('div');box.className='source-box';
 const btn=document.createElement('button');btn.className='source-toggle';btn.type='button';
 btn.innerHTML=`<span>▤ &nbsp; Sources (${sources.length})</span><span>⌄</span>`;
 const list=document.createElement('div');list.className='source-list';list.hidden=true;
 btn.onclick=()=>{list.hidden=!list.hidden;btn.lastElementChild.textContent=list.hidden?'⌄':'⌃'};
 sources.forEach((s,i)=>{
  const item=document.createElement('div');item.className='source-item';item.dataset.sourceIndex=String(i+1);
  const name=document.createElement('div');name.className='source-name';name.textContent=`[Source ${i+1}] ${s.source||'Unknown source'}`;
  const text=document.createElement('div');text.className='source-text';text.textContent=s.content||'';
  const meta=document.createElement('div');meta.className='source-meta';
  const dist=s.distance==null?'':` · vector distance ${Number(s.distance).toFixed(3)}`;
  const rel=s.relevance==null?'':` · score ${Number(s.relevance).toFixed(3)}`;
  meta.textContent=`${s.method||'Retrieved'}${dist}${rel}`;
  item.append(name,text,meta);list.appendChild(item);
 });
 box.append(btn,list);return box;
}
function safeMarkdown(text){
 let src=String(text||'').replace(/\r/g,'');
 src=src.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
 src=src.replace(/&lt;br\s*\/?&gt;/gi,'<br>');
 const code=[];src=src.replace(/```([\s\S]*?)```/g,(_,v)=>{code.push(v);return `@@CODE${code.length-1}@@`});
 src=src.replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\*(.*?)\*/g,'<em>$1</em>');
 // Convert inline [Source N] references into clickable citation badges
 src=src.replace(/\[Source\s+(\d+)\]/gi,'<button type="button" class="cite-chip" data-cite="$1">[Source $1]</button>');
 const lines=src.split('\n'),out=[];let ul=false,ol=false,table=false;
 const closeLists=()=>{if(ul){out.push('</ul>');ul=false}if(ol){out.push('</ol>');ol=false}};
 for(let i=0;i<lines.length;i++){let line=lines[i];
  if(/^\s*\|.*\|\s*$/.test(line)){closeLists();if(/^\s*\|?\s*:?-{3,}/.test(line))continue;const cells=line.trim().replace(/^\||\|$/g,'').split('|').map(x=>x.trim());if(!table){out.push('<table><tbody>');table=true}out.push('<tr>'+cells.map(c=>`<td>${c}</td>`).join('')+'</tr>');continue}else if(table){out.push('</tbody></table>');table=false}
  let m;if((m=line.match(/^(#{1,3})\s+(.+)$/))){closeLists();out.push(`<h${m[1].length}>${m[2]}</h${m[1].length}>`)}
  else if((m=line.match(/^\s*[-*]\s+(.+)$/))){if(ol){out.push('</ol>');ol=false}if(!ul){out.push('<ul>');ul=true}out.push(`<li>${m[1]}</li>`)}
  else if((m=line.match(/^\s*\d+[.)]\s+(.+)$/))){if(ul){out.push('</ul>');ul=false}if(!ol){out.push('<ol>');ol=true}out.push(`<li>${m[1]}</li>`)}
  else {closeLists();if(!line.trim())out.push('<div class="md-spacer"></div>');else out.push(`<p>${line}</p>`)}
 }
 closeLists();if(table)out.push('</tbody></table>');let html=out.join('');html=html.replace(/@@CODE(\d+)@@/g,(_,i)=>`<pre><code>${code[Number(i)]}</code></pre>`);return html;
}
function bindCitationClicks(container){
 container.querySelectorAll('.cite-chip').forEach(chip=>{
  chip.onclick=()=>{
   const idx=chip.dataset.cite;
   const host=chip.closest('.message-body')||chip.closest('.result-panel')||container;
   const box=host?.querySelector('.source-box');
   if(!box)return;
   const list=box.querySelector('.source-list');
   const btn=box.querySelector('.source-toggle');
   if(list&&list.hidden){list.hidden=false;if(btn&&btn.lastElementChild)btn.lastElementChild.textContent='⌃';}
   const target=box.querySelector(`.source-item[data-source-index="${idx}"]`);
   if(target){
    target.scrollIntoView({behavior:'smooth',block:'nearest'});
    target.classList.add('cite-highlight');
    setTimeout(()=>target.classList.remove('cite-highlight'),2200);
   }
  };
 });
}
function scrollChatBottom(){const region=$('chatScrollRegion')||$('chatMessages');if(region)region.scrollTop=region.scrollHeight;$('chatMessages').scrollTop=$('chatMessages').scrollHeight}
function addMessage(message){
 $('welcome').style.display='none';
 const wrap=document.createElement('div');wrap.className=`message ${message.role}`;
 const avatar=document.createElement('div');avatar.className='avatar';avatar.textContent=message.role==='user'?'Y':'R';
 const body=document.createElement('div');body.className='message-body';
 const role=document.createElement('div');role.className='message-role';role.textContent=message.role==='user'?'YOU':'RAG ASSISTANT';
 const content=document.createElement('div');content.className='message-content';content.innerHTML=safeMarkdown(message.content||'');
 body.append(role,content);
 if(message.grounding){
  const gBadge=makeGroundingBadge(message.grounding);
  if(gBadge)body.appendChild(gBadge);
 }
 const src=makeSourceBox(message.sources);if(src)body.appendChild(src);
 if(message.role==='assistant' && message.content){
  const msgFoot=document.createElement('div');msgFoot.className='msg-footer-bar';
  const copyMsgBtn=document.createElement('button');copyMsgBtn.type='button';copyMsgBtn.className='text-button msg-copy-btn';copyMsgBtn.textContent='Copy';
  copyMsgBtn.onclick=async()=>{try{await navigator.clipboard.writeText(String(message.content||''));copyMsgBtn.textContent='Copied ✓';setTimeout(()=>copyMsgBtn.textContent='Copy',1800)}catch{toast('Could not copy message',true)}};
  msgFoot.appendChild(copyMsgBtn);
  if(message.latency_seconds){const meta=document.createElement('span');meta.className='source-meta';meta.textContent=`Generated in ${message.latency_seconds}s`;msgFoot.appendChild(meta)}
  body.appendChild(msgFoot);
 }
 bindCitationClicks(body);
 wrap.append(avatar,body);$('chatMessages').appendChild(wrap);scrollChatBottom();
}
function renderMessages(messages){$('chatMessages').innerHTML='';currentMessages=messages||[];if(!currentMessages.length){$('welcome').style.display='block';return}$('welcome').style.display='none';currentMessages.forEach(addMessage)}
async function refreshSessions(){await renderHistory()}
async function openSession(id){try{const s=await api(`/api/sessions/${encodeURIComponent(id)}`);currentSession=s.id;selectedSourceIds=new Set(s.source_ids||[]);updateSelectedSourcesBadge();renderMessages(s.messages||[]);await refreshDocuments();await refreshSessions();setView('chat')}catch(e){toast(e.message,true)}}
function newChat(){currentSession=null;selectedSourceIds=new Set();updateSelectedSourcesBadge();renderMessages([]);refreshSessions();refreshDocuments();setView('chat');$('questionInput').focus()}
async function sendQuestion(){const question=$('questionInput').value.trim();if(!question||busy)return;if(!selectedSourceIds.size){toast('Select at least one indexed source in the Knowledge Base before sending.',true);return;}$('questionInput').value='';$('questionInput').style.height='auto';if(!currentSession)currentSession=crypto.randomUUID();addMessage({role:'user',content:question});const placeholder=document.createElement('div');placeholder.className='message assistant';placeholder.id='thinkingMessage';placeholder.innerHTML='<div class="avatar">R</div><div class="message-body"><div class="message-role">RAG ASSISTANT</div><div class="message-content">Searching evidence and preparing an answer…</div></div>';$('chatMessages').appendChild(placeholder);scrollChatBottom();setBusy(true);try{const data=await api('/api/chat',{method:'POST',body:JSON.stringify({question,session_id:currentSession,use_reranker:$('rerankToggle').checked,top_k:5,source_ids:[...selectedSourceIds]})});currentSession=data.session_id;$('thinkingMessage')?.remove();addMessage({role:'assistant',content:data.answer,sources:data.sources,grounding:data.grounding,latency_seconds:data.latency_seconds});currentMessages=data.session.messages||[];await refreshSessions()}catch(e){$('thinkingMessage')?.remove();addMessage({role:'assistant',content:`I couldn't complete that request.\n\n${e.message}`});toast(e.message,true)}finally{setBusy(false)}}
async function openVersionsModal(docId, docName){
 try{
  const data=await api(`/api/documents/${encodeURIComponent(docId)}/versions`);
  $('versionsModalTitle').textContent=`Version history · ${data.name||docName}`;
  const host=$('versionsContent');host.innerHTML='';
  const cur=document.createElement('div');cur.className='version-card active-version';
  cur.innerHTML=`<div><b>Current active version: v${data.current.version}</b><div class="doc-meta">${data.current.chunks} chunks · ${data.current.updated_at||'Active'}</div></div>`;
  host.appendChild(cur);
  if(data.latest_diff){
   const diffBox=document.createElement('div');diffBox.className='version-diff-box';
   const h=document.createElement('div');h.className='version-diff-header';
   h.textContent=`Changes from v${data.latest_diff.from_version} → v${data.latest_diff.to_version} (+${data.latest_diff.lines_added} / -${data.latest_diff.lines_removed} lines)`;
   const pre=document.createElement('pre');pre.className='version-diff-pre';pre.textContent=data.latest_diff.diff_preview;
   diffBox.append(h,pre);host.appendChild(diffBox);
  }
  if(!data.archived_versions.length){
   const empty=document.createElement('div');empty.className='empty-docs';empty.textContent='No older archived versions yet. Uploading a modified file with the same filename will automatically archive v1 here.';
   host.appendChild(empty);
  }else{
   data.archived_versions.forEach(v=>{
    const row=document.createElement('div');row.className='version-card';
    const info=document.createElement('div');
    info.innerHTML=`<b>Version v${v.version}</b><div class="doc-meta">${v.chunks||0} chunks · Archived ${v.archived_at||''}</div>`;
    const restoreBtn=document.createElement('button');restoreBtn.type='button';restoreBtn.className='outline-button';restoreBtn.textContent=`Restore v${v.version}`;
    restoreBtn.onclick=async()=>{
     if(!confirm(`Restore version v${v.version} of "${docName}"? Current version will be archived safely.`))return;
     restoreBtn.disabled=true;
     try{
      const res=await api(`/api/documents/${encodeURIComponent(docId)}/versions/${v.version}/restore`,{method:'POST'});
      toast(`Restored v${v.version} as active v${res.new_version}.`);
      await refreshDocuments();
      await openVersionsModal(docId, docName);
     }catch(err){toast(err.message,true);restoreBtn.disabled=false;}
    };
    row.append(info,restoreBtn);host.appendChild(row);
   });
  }
  $('versionsModal').classList.remove('hidden');
 }catch(e){toast(e.message,true)}
}
async function refreshDocuments(){
 try{
  const data=await api('/api/documents');
  $('docCount').textContent=`${data.count} source${data.count===1?'':'s'} in library`;
  $('docCountSmall').textContent=data.count;
  if($('drawerCountBadge'))$('drawerCountBadge').textContent=data.count;
  if($('clearAllDocs'))$('clearAllDocs').disabled=data.count===0;
  const list=$('documentList'), sourceSelect=$('studySource'), quizSource=$('quizSource');
  const selected=sourceSelect.value, qselected=quizSource?.value||'all';
  sourceSelect.innerHTML='<option value="all">All indexed documents</option>';
  if(quizSource)quizSource.innerHTML='<option value="all">All indexed documents</option>';
  if($('compareDocs'))$('compareDocs').innerHTML='';
  list.innerHTML='';
  data.documents.forEach(d=>{
   const id=d.id||d.doc_id||d.name;
   const opt=document.createElement('option');opt.value=id;opt.textContent=d.name;sourceSelect.appendChild(opt);
   if(quizSource){const qo=document.createElement('option');qo.value=id;qo.textContent=d.name;quizSource.appendChild(qo)}
   if($('compareDocs')){const o=document.createElement('option');o.value=id;o.textContent=d.name+' · '+String(id).slice(0,6);o.dataset.name=d.name;$('compareDocs').appendChild(o)}
   const row=document.createElement('div');row.className='doc-row';
   const isWeb=String((d.type||'').toLowerCase().includes('web')||String(d.name||'').startsWith('http'));
   row.dataset.web=String(isWeb);
   const pick=document.createElement('input');pick.type='checkbox';pick.className='source-pick';pick.checked=selectedSourceIds.has(id);pick.disabled=!d.indexed;pick.title=d.indexed?'Use this source in the current chat':'This source is listed in the library but its vector index is unavailable';pick.setAttribute('aria-label',`Use ${d.name} in this chat`);
   pick.onchange=async()=>{if(pick.checked)selectedSourceIds.add(id);else selectedSourceIds.delete(id);updateSelectedSourcesBadge();if(currentSession){try{await api(`/api/sessions/${encodeURIComponent(currentSession)}/sources`,{method:'PUT',body:JSON.stringify({source_ids:[...selectedSourceIds]})})}catch(e){toast(e.message,true)}}};
   const icon=document.createElement('div');icon.className='file-icon';icon.textContent=(d.type||'doc').slice(0,3).toUpperCase();
   const info=document.createElement('div');info.className='doc-info';
   const name=document.createElement('div');name.className='doc-name';name.textContent=d.name;name.title=d.name;
   const verText=d.version&&d.version>1?` · v${d.version}`:'';
   const meta=document.createElement('div');meta.className='doc-meta';meta.textContent=`${d.chunks||0} chunks · ${d.size==null?'web page':formatBytes(d.size)}${d.indexed?' · indexed':' · not indexed'}${verText}`;
   info.append(name,meta);
   if(!isWeb||isWeb==='false'){
    const verBtn=document.createElement('button');verBtn.className='text-button doc-ver-btn';verBtn.type='button';
    verBtn.textContent=`v${d.version||1}`;verBtn.title='View version history or restore an older revision';
    verBtn.onclick=()=>openVersionsModal(id,d.name);
    row.append(pick,icon,info,verBtn);
   }else{
    row.append(pick,icon,info);
   }
   const del=document.createElement('button');del.className='doc-delete';del.type='button';del.title='Remove source from library';del.textContent='×';del.onclick=()=>deleteDocument(id,d.name);
   row.appendChild(del);list.appendChild(row);
  });
  if(!data.documents.length)list.innerHTML='<div class="empty-docs">Upload a file or add a web page to get started.</div>';
  if($('homeRecentDocs')){
   const hd=$('homeRecentDocs');hd.innerHTML='';
   if(!data.documents.length){
    hd.innerHTML='<div class="home-mini-empty">No documents indexed yet. Upload files or add a URL to populate your library.</div>';
   }else{
    data.documents.slice(0,4).forEach(d=>{
     const id=d.id||d.doc_id||d.name;
     const btn=document.createElement('button');btn.type='button';btn.className='home-mini-row';
     const b=document.createElement('b');b.textContent=d.name;
     const sm=document.createElement('small');sm.textContent=`${d.chunks||0} chunks · v${d.version||1}`;
     btn.append(b,sm);
     btn.onclick=()=>{if(d.indexed)selectedSourceIds.add(id);updateSelectedSourcesBadge();setView('chat');refreshDocuments()};
     hd.appendChild(btn);
    });
   }
  }
  if([...sourceSelect.options].some(o=>o.value===selected))sourceSelect.value=selected;
  if(quizSource&&[...quizSource.options].some(o=>o.value===qselected))quizSource.value=qselected;
  knownDocsById=new Map(data.documents.map(d=>[d.id||d.doc_id||d.name, d.name||'Source']));
  const indexedDocs=data.documents.filter(d=>d.indexed);
  const valid=new Set(indexedDocs.map(d=>d.id||d.doc_id||d.name));
  selectedSourceIds=new Set([...selectedSourceIds].filter(id=>valid.has(id)));
  if(selectedSourceIds.size===0 && indexedDocs.length===1 && !currentSession){
   const onlyId=indexedDocs[0].id||indexedDocs[0].doc_id||indexedDocs[0].name;
   selectedSourceIds.add(onlyId);
  }
  [...list.querySelectorAll('.doc-row')].forEach((row,i)=>{const d=data.documents[i],id=d.id||d.doc_id||d.name;row.querySelector('.source-pick').checked=selectedSourceIds.has(id)});
  updateSelectedSourcesBadge();
 }catch(e){toast(`Could not load documents: ${e.message}`,true)}
}
async function deleteDocument(id,name){if(!confirm(`Remove “${name}” from the knowledge base? This also deletes the local uploaded file if it is one.`))return;try{const d=await api(`/api/documents/${encodeURIComponent(id)}`,{method:'DELETE'});await refreshDocuments();toast(`Removed ${name}; ${d.chunks_removed} chunks deleted.`)}catch(e){toast(e.message,true)}}
async function clearAllDocuments(){
 if(!confirm('Clear ALL documents, websites, version archives, and indexed vectors from your Knowledge Base? Your conversation history will be kept, but all uploaded sources will be permanently removed.'))return;
 setBusy(true);
 try{
  const d=await api('/api/documents',{method:'DELETE'});
  selectedSourceIds.clear();
  updateSelectedSourcesBadge();
  await refreshDocuments();
  await checkHealth();
  toast(`Cleared ${d.sources_removed||0} source(s) and ${d.chunks_removed||0} vector chunks.`);
 }catch(e){toast(e.message,true)}finally{setBusy(false)}
}
async function syncSessionSourcesIfNeeded(){
 if(currentSession){
  try{await api(`/api/sessions/${encodeURIComponent(currentSession)}/sources`,{method:'PUT',body:JSON.stringify({source_ids:[...selectedSourceIds]})})}catch{}
 }
}
async function uploadFiles(files){
 if(!files.length)return;
 const fd=new FormData();files.forEach(f=>fd.append('files',f));
 $('uploadStatus').textContent=`Uploading and indexing ${files.length} file(s)…`;setBusy(true);
 try{
  const data=await api('/api/upload',{method:'POST',body:fd});
  const lines=data.results.map(r=>r.status==='indexed'?`✓ ${r.name}: indexed ${r.chunks} chunks`:r.status==='updated'?`↻ ${r.name}: updated to v${r.version} (${r.chunks} chunks)`:r.status==='duplicate'?`↻ ${r.name}: already in library; reused existing index`:`✕ ${r.name}: ${r.error}`);
  $('uploadStatus').textContent=lines.join(' · ');
  let autoPicked=false;
  data.results.forEach(r=>{
   if(r.status==='error'){toast(`${r.name}: ${r.error}`,true);return;}
   if(r.id && selectedSourceIds.size===0){
    selectedSourceIds.add(r.id);
    autoPicked=true;
   }
   if(r.status==='duplicate')toast(`${r.name} already exists; reused existing index`);
   else if(r.status==='updated')toast(r.message||`Updated ${r.name} to v${r.version}`);
   else toast(`Indexed ${r.name} (${r.chunks} chunks)`);
  });
  const prevCount=knownDocsById.size;
  await refreshDocuments();
  if(autoPicked)await syncSessionSourcesIfNeeded();
  if(prevCount>=1 && knownDocsById.size>prevCount){$('uploadModal')?.classList.add('hidden');setKnowledgePopoverOpen(true);}
 }catch(e){$('uploadStatus').textContent=e.message;toast(e.message,true)}
 finally{setBusy(false);$('fileInput').value=''}
}
async function addUrl(){
 const url=$('urlInput').value.trim();if(!url)return toast('Enter a complete web URL first.',true);
 $('uploadStatus').textContent='Fetching and indexing web page…';setBusy(true);
 try{
  const fd=new FormData();fd.append('url',url);
  const data=await api('/api/url',{method:'POST',body:fd});
  $('uploadStatus').textContent=data.status==='duplicate'?`↻ Already in library: ${data.name} (${data.chunks} chunks reused)`:`✓ Indexed ${data.name}: ${data.chunks} chunks`;
  $('urlInput').value='';
  let autoPicked=false;
  if(data.id && selectedSourceIds.size===0){
   selectedSourceIds.add(data.id);
   autoPicked=true;
  }
  toast(data.status==='duplicate'?'Website already exists; reused existing index':'Web page indexed');
  const prevCount=knownDocsById.size;
  await refreshDocuments();
  if(autoPicked)await syncSessionSourcesIfNeeded();
  if(prevCount>=1 && knownDocsById.size>prevCount){$('uploadModal')?.classList.add('hidden');setKnowledgePopoverOpen(true);}
 }catch(e){$('uploadStatus').textContent=e.message;toast(e.message,true)}
 finally{setBusy(false)}
}
function showResult(id,text,sources,coverage,grounding=null){
 if(id==='researchResult')$('researchPlaceholder')?.classList.add('hidden');
 if(id==='studyResult')$('studyPlaceholder')?.classList.add('hidden');
 if(id==='quizResult')$('quizPlaceholder')?.classList.add('hidden');
 const panel=$(id);panel.classList.remove('hidden');
 const rawText=String(text||'');
 panel.innerHTML='';
 // Action toolbar for Copy and Export .md
 const toolbar=document.createElement('div');toolbar.className='result-toolbar';
 if(grounding){
  const gb=makeGroundingBadge(grounding);
  if(gb)toolbar.appendChild(gb);
 }else{
  toolbar.appendChild(document.createElement('span'));
 }
 const actions=document.createElement('div');actions.className='result-actions';
 const copyBtn=document.createElement('button');copyBtn.type='button';copyBtn.className='text-button';copyBtn.textContent='Copy markdown';
 copyBtn.onclick=async()=>{try{await navigator.clipboard.writeText(rawText);toast('Copied to clipboard')}catch{toast('Could not copy text',true)}};
 const exportBtn=document.createElement('button');exportBtn.type='button';exportBtn.className='text-button';exportBtn.textContent='↓ Export .md';
 exportBtn.onclick=()=>{const blob=new Blob([rawText],{type:'text/markdown;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`rag-studio-${id}-${new Date().toISOString().slice(0,10)}.md`;a.click();URL.revokeObjectURL(a.href)};
 actions.append(copyBtn,exportBtn);toolbar.appendChild(actions);
 const body=document.createElement('div');body.className='result-body';body.innerHTML=safeMarkdown(rawText);
 panel.append(toolbar,body);
 if(Array.isArray(coverage)&&coverage.length){
  const cov=document.createElement('div');cov.className='result-coverage';
  cov.textContent='Document coverage: '+coverage.map(c=>`${c.name} (${c.passages_retrieved} passage${c.passages_retrieved===1?'':'s'})`).join(' · ');
  panel.appendChild(cov);
 }
 if(sources&&sources.length){
  const srcBox=makeSourceBox(sources);
  if(srcBox)panel.appendChild(srcBox);
  const line=document.createElement('div');line.className='result-sources';line.textContent='Sources: '+[...new Set(sources.map(s=>s.source).filter(Boolean))].join(' · ');panel.appendChild(line);
 }
 bindCitationClicks(panel);
}
async function runResearch(){const task=$('researchTask').value.trim();if(!task)return toast('Enter a research task.',true);setBusy(true);$('runResearch').textContent='Researching…';$('researchPlaceholder')?.classList.add('hidden');$('researchResult').classList.remove('hidden');$('researchResult').textContent='Retrieving and synthesizing evidence…';try{const chosen=[...($('compareDocs')?.selectedOptions||[])].map(o=>o.value).filter(Boolean);if(chosen.length>5){toast('Choose no more than five documents.',true);setBusy(false);return}const data=await api('/api/research',{method:'POST',body:JSON.stringify({task,use_reranker:$('researchRerank').checked,sources:chosen})});showResult('researchResult',data.result,data.sources,data.coverage,data.grounding);toast('Research complete')}catch(e){showResult('researchResult',e.message);toast(e.message,true)}finally{setBusy(false);$('runResearch').textContent='Run research ↗'}}
async function generateStudy(){const kind=$('studyKind').value;setBusy(true);$('generateStudy').textContent='Generating…';$('studyPlaceholder')?.classList.add('hidden');$('studyResult').classList.remove('hidden');$('studyResult').textContent='Building study material from retrieved passages…';try{const data=await api('/api/study',{method:'POST',body:JSON.stringify({source:$('studySource').value,kind,count:Number($('studyCount').value)})});showResult('studyResult',data.result||'No study material was returned.',data.sources);toast('Study material ready')}catch(e){showResult('studyResult',e.message);toast(e.message,true)}finally{setBusy(false);$('generateStudy').textContent='Generate study material ↗'}}
async function generateQuiz(){setBusy(true);$('generateQuiz').textContent='Generating…';$('quizPlaceholder')?.classList.add('hidden');$('quizResult').classList.remove('hidden');$('quizResult').textContent='Preparing questions and answer choices…';try{const data=await api('/api/study',{method:'POST',body:JSON.stringify({source:$('quizSource').value,kind:'quiz',count:Number($('quizCount').value)})});let quiz=Array.isArray(data.quiz)?data.quiz:parseQuiz(data.result);if(quiz&&quiz.length){renderQuiz(quiz,data.sources,'quizResult',data.warning)}else{showResult('quizResult','The server returned no usable quiz. Check the terminal for the validation reason, then try a smaller question count.');}toast(data.warning||'Quiz ready')}catch(e){showResult('quizResult',e.message);toast(e.message,true)}finally{setBusy(false);$('generateQuiz').textContent='Generate quiz ↗'}}
function parseQuiz(raw){let t=String(raw||'').trim().replace(/^```(?:json)?\s*/i,'').replace(/\s*```$/,'');const first=t.indexOf('{'),last=t.lastIndexOf('}');if(first>=0&&last>first)t=t.slice(first,last+1);try{const d=JSON.parse(t);const qs=Array.isArray(d)?d:d.questions; if(!Array.isArray(qs))return null;const valid=qs.map(q=>({question:String(q.question||''),options:Array.isArray(q.options)?q.options.map(String):[],answer_index:Number(q.answer_index),explanation:String(q.explanation||''),source:String(q.source||'')})).filter(q=>q.question&&q.options.length===4&&Number.isInteger(q.answer_index)&&q.answer_index>=0&&q.answer_index<4);return valid.length?valid:null}catch{return null}}
function renderQuiz(questions,sources,targetId='quizResult',warning=null){
 const host=$(targetId);host.innerHTML='';let current=0,submitted=false;const answers=Array(questions.length).fill(null);
 const title=document.createElement('h3');title.textContent='Practice quiz';host.appendChild(title);
 const intro=document.createElement('p');intro.textContent=`${questions.length} questions · Choose an answer for each, then submit to see your score.`;host.appendChild(intro);
 if(warning){const warnEl=document.createElement('div');warnEl.className='quiz-progress';warnEl.textContent=warning;host.appendChild(warnEl);}
 const progress=document.createElement('div');progress.className='quiz-progress';host.appendChild(progress);
 const card=document.createElement('fieldset');card.className='quiz-question quiz-single';host.appendChild(card);
 const nav=document.createElement('div');nav.className='quiz-navigation';const prev=document.createElement('button');prev.type='button';prev.className='outline-button';prev.textContent='← Previous';const next=document.createElement('button');next.type='button';next.className='primary-button';next.textContent='Next question →';nav.append(prev,next);host.appendChild(nav);
 const submit=document.createElement('button');submit.type='button';submit.className='primary-button spaced-button';submit.textContent='Submit quiz';host.appendChild(submit);
 const score=document.createElement('div');score.className='quiz-score';score.hidden=true;host.appendChild(score);
 function paint(){const q=questions[current];card.innerHTML='';const legend=document.createElement('legend');legend.textContent=`Question ${current+1} of ${questions.length}: ${q.question||'Question'}`;card.appendChild(legend);(q.options||[]).forEach((opt,j)=>{const label=document.createElement('label');label.className='quiz-option';const radio=document.createElement('input');radio.type='radio';radio.name='currentQuizAnswer';radio.value=String(j);radio.checked=answers[current]===j;radio.disabled=submitted;radio.onchange=()=>{answers[current]=j;paintProgress()};label.append(radio,document.createTextNode(String(opt)));card.appendChild(label)});progress.textContent=`Progress: ${answers.filter(x=>x!==null).length}/${questions.length} answered`;prev.disabled=current===0;next.hidden=current===questions.length-1||submitted;submit.hidden=submitted;}
 function paintProgress(){progress.textContent=`Question ${current+1}/${questions.length} · ${answers.filter(x=>x!==null).length} answered`}
 prev.onclick=()=>{if(current>0){current--;paint()}};next.onclick=()=>{if(current<questions.length-1){current++;paint()}};
 submit.onclick=()=>{if(answers.some(x=>x===null)){if(!confirm(`You have ${answers.filter(x=>x===null).length} unanswered question(s). Submit anyway?`))return}submitted=true;let total=0;const review=document.createElement('div');review.className='quiz-review';questions.forEach((q,i)=>{const correct=Number(q.answer_index);if(answers[i]===correct)total++;const item=document.createElement('div');item.className='quiz-review-item';const heading=document.createElement('b');heading.textContent=`${i+1}. ${q.question}`;const detail=document.createElement('p');detail.textContent=`${answers[i]===correct?'Correct':'Incorrect'} · Correct answer: ${q.options[correct]||'Unavailable'}. ${q.explanation||''} ${q.source||''}`;item.append(heading,detail);review.appendChild(item)});score.hidden=false;score.textContent=`Your score: ${total}/${questions.length} (${Math.round(total/questions.length*100)}%)`;host.appendChild(review);submit.disabled=true;next.hidden=true;prev.disabled=true;paint()};
 paint();if(sources?.length){const note=document.createElement('div');note.className='result-sources';note.textContent='Generated from: '+[...new Set(sources.map(s=>s.source))].join(' · ');host.appendChild(note)}
}
async function renderHistory(){
 try{
  const d=await api('/api/sessions');
  const host=$('historyList');if(!host)return;host.innerHTML='';
  if($('homeRecentChats')){
   const hc=$('homeRecentChats');hc.innerHTML='';
   if(!d.sessions.length){
    hc.innerHTML='<div class="home-mini-empty">No saved conversations yet. Ask a question above to start your first session.</div>';
   }else{
    d.sessions.slice(0,4).forEach(s=>{
     const btn=document.createElement('button');btn.type='button';btn.className='home-mini-row';
     const b=document.createElement('b');b.textContent=s.title||'Conversation';
     const sm=document.createElement('small');sm.textContent=(s.updated_at||s.created_at||'').slice(0,16).replace('T',' ');
     btn.append(b,sm);
     btn.onclick=()=>openSession(s.id);
     hc.appendChild(btn);
    });
   }
  }
  if(!d.sessions.length){host.innerHTML='<div class="empty-docs">No saved conversations yet. Click "＋ New conversation" to begin.</div>';return}
  d.sessions.forEach(s=>{
   const row=document.createElement('div');row.className='history-row';
   const open=document.createElement('button');open.className='history-open';open.type='button';open.textContent=s.title||'Conversation';open.onclick=()=>openSession(s.id);
   const meta=document.createElement('span');meta.textContent=s.updated_at||s.created_at||'';
   const actions=document.createElement('div');actions.className='history-actions';
   const ren=document.createElement('button');ren.className='outline-button';ren.type='button';ren.textContent='Rename';
   ren.onclick=async()=>{
    const nextTitle=prompt('Rename conversation:',s.title||'Conversation');
    if(!nextTitle||!nextTitle.trim())return;
    try{
     await api(`/api/sessions/${encodeURIComponent(s.id)}`,{method:'PATCH',body:JSON.stringify({title:nextTitle.trim()})});
     await renderHistory();toast('Conversation renamed');
    }catch(e){toast(e.message,true)}
   };
   const del=document.createElement('button');del.className='outline-button';del.type='button';del.textContent='Delete';
   del.onclick=async()=>{
    if(!confirm(`Delete “${s.title||'Conversation'}”?`))return;
    try{
     await api(`/api/sessions/${encodeURIComponent(s.id)}`,{method:'DELETE'});
     if(currentSession===s.id){currentSession=null;selectedSourceIds=new Set();updateSelectedSourcesBadge();renderMessages([])}
     await renderHistory();toast('Conversation deleted');
    }catch(e){toast(e.message,true)}
   };
   actions.append(ren,del);
   row.append(open,meta,actions);host.appendChild(row);
  });
 }catch(e){toast('Could not load history: '+e.message,true)}
}
function startHomeQuestion(){const q=$('homeQuestion')?.value.trim();if(!q)return;setView('chat');$('questionInput').value=q;$('homeQuestion').value='';if(!selectedSourceIds.size){toast('Choose one or more indexed sources in the Knowledge Base, then send your question.',true);$('questionInput').focus();return;}sendQuestion()}
async function checkHealth(){
 try{
  const d=await api('/api/health');
  $('statusDot').classList.toggle('ok',d.ok);
  $('statusText').textContent=d.ok?'Local service ready':'Service unavailable';
  $('modelInfo').textContent=`${d.model}`;
  if($('firstLaunchBanner'))$('firstLaunchBanner').classList.toggle('hidden',Boolean(d.groq_configured));
  if(!d.groq_configured){
   if(!hasPromptedKeySetup){
    hasPromptedKeySetup=true;
    if($('groqApiKey'))$('groqApiKey').value='';
    $('settingsModal')?.classList.remove('hidden');
    refreshKeyStatus();
    setTimeout(()=>$('groqApiKey')?.focus(),120);
   }
  }
  $('settingsDetails').innerHTML='';
  [['Service','Running locally (v1.0)'],['Groq API key',d.groq_configured?'Configured':'Not configured'],['Generation model',d.model],['Embedding model',d.embedding_model],['Indexed sources',String(d.documents)]].forEach(([k,v])=>{
   const line=document.createElement('div');line.className='setting-line';
   const a=document.createElement('span'),b=document.createElement('span');a.textContent=k;b.textContent=v;
   line.append(a,b);$('settingsDetails').appendChild(line);
  });
 }catch(e){$('statusText').textContent='Service unavailable';$('modelInfo').textContent='API not responding';$('statusDot').classList.remove('ok')}
}
document.querySelectorAll('.nav-item').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
function setKnowledgePopoverOpen(open){
 const drawer=$('knowledgeDrawer');
 const btn=$('toggleDrawerBtn');
 if(!drawer)return;
 drawer.classList.toggle('hidden',!open);
 if(btn)btn.setAttribute('aria-expanded',String(open));
}
if($('toggleDrawerBtn'))$('toggleDrawerBtn').onclick=(e)=>{e.stopPropagation();const drawer=$('knowledgeDrawer');if(drawer)setKnowledgePopoverOpen(drawer.classList.contains('hidden'))};
if($('closeDrawerBtn'))$('closeDrawerBtn').onclick=()=>setKnowledgePopoverOpen(false);
if($('selectedSourcesBadge'))$('selectedSourcesBadge').onclick=(e)=>{e.stopPropagation();const drawer=$('knowledgeDrawer');if(drawer)setKnowledgePopoverOpen(drawer.classList.contains('hidden'))};
document.addEventListener('click',(e)=>{
 const drawer=$('knowledgeDrawer');
 if(!drawer||drawer.classList.contains('hidden'))return;
 if(drawer.contains(e.target))return;
 if($('toggleDrawerBtn')?.contains(e.target)||$('selectedSourcesBadge')?.contains(e.target))return;
 setKnowledgePopoverOpen(false);
});
document.addEventListener('keydown',(e)=>{if(e.key==='Escape')setKnowledgePopoverOpen(false)});
if($('clearHistory'))$('clearHistory').onclick=async()=>{if(!confirm('Clear all saved conversations? Your documents and indexed vectors will not be deleted.'))return;try{const d=await api('/api/sessions',{method:'DELETE'});currentSession=null;selectedSourceIds=new Set();updateSelectedSourcesBadge();renderMessages([]);await refreshSessions();setView('home');toast(`Cleared ${d.deleted_count} conversation(s).`)}catch(e){toast(e.message,true)}};
if($('homeAsk'))$('homeAsk').onclick=startHomeQuestion;if($('homeQuestion'))$('homeQuestion').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();startHomeQuestion()}});document.querySelectorAll('[data-home-action]').forEach(b=>b.onclick=()=>{const a=b.dataset.homeAction;if(a==='upload')openUploadModal();else if(a==='website')openUploadModal(true);else setView(a==='research'?'research':'study')});$('newChat').onclick=newChat;$('sendButton').onclick=sendQuestion;$('questionInput').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendQuestion()}});$('questionInput').addEventListener('input',e=>{e.target.style.height='auto';e.target.style.height=Math.min(e.target.scrollHeight,120)+'px'});document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('questionInput').value=b.dataset.prompt;$('questionInput').focus()});$('chooseFiles').onclick=()=>$('fileInput').click();$('openUpload').onclick=()=>openUploadModal();$('openWebsite').onclick=()=>openUploadModal(true);$('closeUpload').onclick=()=> $('uploadModal').classList.add('hidden');$('uploadModal').addEventListener('click',e=>{if(e.target===$('uploadModal'))$('uploadModal').classList.add('hidden')});if($('closeVersions'))$('closeVersions').onclick=()=>$('versionsModal').classList.add('hidden');if($('versionsModal'))$('versionsModal').addEventListener('click',e=>{if(e.target===$('versionsModal'))$('versionsModal').classList.add('hidden')});$('fileInput').onchange=e=>uploadFiles([...e.target.files]);$('refreshDocs').onclick=refreshDocuments;if($('clearAllDocs'))$('clearAllDocs').onclick=clearAllDocuments;$('addUrl').onclick=addUrl;$('urlInput').addEventListener('keydown',e=>{if(e.key==='Enter')addUrl()});$('runResearch').onclick=runResearch;$('generateStudy').onclick=generateStudy;$('generateQuiz').onclick=generateQuiz;$('settingsButton').onclick=()=>{$('settingsModal').classList.remove('hidden');checkHealth();refreshKeyStatus()};
if($('themesButton'))$('themesButton').onclick=()=>$('themesModal')?.classList.remove('hidden');
if($('closeThemes'))$('closeThemes').onclick=()=>$('themesModal')?.classList.add('hidden');
if($('closeThemesBottom'))$('closeThemesBottom').onclick=()=>$('themesModal')?.classList.add('hidden');
if($('themesModal'))$('themesModal').addEventListener('click',e=>{if(e.target===$('themesModal'))$('themesModal').classList.add('hidden')});$('closeSettings').onclick=$('closeSettingsBottom').onclick=()=>$('settingsModal').classList.add('hidden');$('settingsModal').addEventListener('click',e=>{if(e.target===$('settingsModal'))$('settingsModal').classList.add('hidden')});document.querySelectorAll('[data-study-view]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-study-view]').forEach(x=>x.classList.toggle('active',x===b));$('studyMaterialPane').classList.toggle('hidden',b.dataset.studyView!=='material');$('studyQuizPane').classList.toggle('hidden',b.dataset.studyView!=='quiz')});document.querySelectorAll('[data-library]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-library]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.doc-row').forEach(r=>{const isWeb=r.dataset.web==='true';r.hidden=b.dataset.library==='documents'&&isWeb||b.dataset.library==='websites'&&!isWeb})});$('dropZone').addEventListener('dragover',e=>{e.preventDefault();$('dropZone').style.background='var(--mint)'});$('dropZone').addEventListener('dragleave',()=>$('dropZone').style.background='');$('dropZone').addEventListener('drop',e=>{e.preventDefault();$('dropZone').style.background='';uploadFiles([...e.dataTransfer.files].filter(f=>/\.(pdf|txt|md|docx|xlsx|xls|pptx|csv|png|jpg|jpeg|webp)$/i.test(f.name)))});
async function refreshKeyStatus(){try{const d=await api('/api/settings/key');$('keyStatus').textContent=d.configured?`Groq key is configured (${d.source||'saved'}).`:'No Groq key saved.';$('removeGroqKey').disabled=!d.configured}catch(e){$('keyStatus').textContent='Could not check key status.'}}
async function saveGroqKey(){const key=$('groqApiKey').value.trim();if(!key)return toast('Paste your Groq API key first.',true);$('saveGroqKey').disabled=true;try{const d=await api('/api/settings/key',{method:'PUT',body:JSON.stringify({api_key:key})});$('groqApiKey').value='';toast(d.message||'Groq key saved securely.');await refreshKeyStatus();await checkHealth()}catch(e){toast(e.message,true)}finally{$('saveGroqKey').disabled=false}}
async function removeGroqKey(){if(!confirm('Remove the saved Groq API key from this computer?'))return;try{const d=await api('/api/settings/key',{method:'DELETE'});$('groqApiKey').value='';await refreshKeyStatus();await checkHealth();toast(d.message||'Saved Groq key removed.')}catch(e){toast(e.message,true)}}
if($('saveGroqKey'))$('saveGroqKey').onclick=saveGroqKey;if($('removeGroqKey'))$('removeGroqKey').onclick=removeGroqKey;
if($('restoreBackupBtn')&&$('restoreFileInput')){
 $('restoreBackupBtn').onclick=()=>$('restoreFileInput').click();
 $('restoreFileInput').onchange=async(e)=>{
  const f=e.target.files?.[0];if(!f)return;
  if(!confirm('Restore workspace from this backup ZIP? Current sessions, documents, versions, and vectors will be replaced (API credentials are preserved).')){e.target.value='';return;}
  $('restoreStatus').textContent='Restoring backup…';$('restoreBackupBtn').disabled=true;
  try{
   const fd=new FormData();fd.append('file',f);
   const res=await api('/api/restore',{method:'POST',body:fd});
   $('restoreStatus').textContent=`Restored (${res.documents_count} docs, ${res.sessions_count} chats).`;
   if(res.preferences?.theme)applyTheme(res.preferences.theme,false);
   currentSession=null;selectedSourceIds=new Set();updateSelectedSourcesBadge();renderMessages([]);
   await refreshDocuments();await refreshSessions();await checkHealth();
   toast(res.message||'Backup restored successfully.');
  }catch(err){$('restoreStatus').textContent='Restore failed.';toast(err.message,true)}
  finally{$('restoreBackupBtn').disabled=false;e.target.value=''}
 };
}
if($('selectAllSources'))$('selectAllSources').onclick=async()=>{try{const d=await api('/api/documents');selectedSourceIds=new Set(d.documents.filter(x=>x.indexed).map(x=>x.id||x.doc_id||x.name));document.querySelectorAll('.source-pick').forEach(el=>{el.checked=!el.disabled});updateSelectedSourcesBadge();if(currentSession)await api(`/api/sessions/${encodeURIComponent(currentSession)}/sources`,{method:'PUT',body:JSON.stringify({source_ids:[...selectedSourceIds]})})}catch(e){toast(e.message,true)}};
if($('clearSourceSelection'))$('clearSourceSelection').onclick=async()=>{selectedSourceIds.clear();document.querySelectorAll('.source-pick').forEach(el=>el.checked=false);updateSelectedSourcesBadge();if(currentSession){try{await api(`/api/sessions/${encodeURIComponent(currentSession)}/sources`,{method:'PUT',body:JSON.stringify({source_ids:[]})})}catch(e){toast(e.message,true)}}};
checkHealth();refreshDocuments();refreshSessions();setView('home');

async function compareSelectedDocuments(){const chosen=[...($('compareDocs')?.selectedOptions||[])];if(chosen.length<2||chosen.length>5)return toast('Select between two and five documents to compare.',true);const names=chosen.map(o=>o.dataset.name||o.textContent);const task=`Compare these selected documents: ${names.map((n,i)=>`Document ${i+1}: ${n}`).join('; ')}. Identify key similarities, differences, potentially conflicting claims, and evidence gaps. Organize findings clearly by document and cite source passages. Do not force a contradiction if evidence does not support one.`;$('researchTask').value=task;setView('research');await runResearch()}
if($('compareDocuments'))$('compareDocuments').onclick=compareSelectedDocuments;
const savedTheme=localStorage.getItem('rag-v3-theme')||'ocean';applyTheme(savedTheme,false);
if($('settingsTheme'))$('settingsTheme').onchange=e=>applyTheme(e.target.value,true);
document.querySelectorAll('[data-theme-pick]').forEach(btn=>btn.onclick=()=>applyTheme(btn.dataset.themePick,true));
document.querySelectorAll('[data-swiss-mode]').forEach(btn=>btn.onclick=()=>applyTheme(btn.dataset.swissMode,true));
function openUploadModal(website=false){$('uploadModal').classList.remove('hidden');if(website)setTimeout(()=>$('urlInput').focus(),80)}

document.querySelectorAll('[data-research-template]').forEach(btn=>{
 btn.onclick=()=>{if($('researchTask')){$('researchTask').value=btn.dataset.researchTemplate;$('researchTask').focus()}};
});
document.querySelectorAll('[data-study-pick]').forEach(card=>{
 card.onclick=()=>{if($('studyKind')){$('studyKind').value=card.dataset.studyPick;generateStudy()}};
});
if($('homeViewAllHistory'))$('homeViewAllHistory').onclick=()=>setView('history');
if($('historyStartChat'))$('historyStartChat').onclick=newChat;

document.addEventListener('keydown',(e)=>{
 if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){
  e.preventDefault();
  const homeActive=$('view-home')?.classList.contains('active');
  if(homeActive&&$('homeQuestion')){$('homeQuestion').focus();return;}
  if($('questionInput')){$('questionInput').focus();}
 }
 if(e.key==='Escape'){
  ['uploadModal','versionsModal','themesModal','settingsModal'].forEach(id=>$(id)?.classList.add('hidden'));
 }
});
