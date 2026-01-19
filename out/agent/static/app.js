const chat = document.getElementById('chat');
const tools = document.getElementById('tools');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function setToolsLog(items) {
  tools.textContent = '';
  for (const it of items) {
    tools.textContent += `# ${it.tool}\n`;
    tools.textContent += `args: ${it.arguments}\n`;
    tools.textContent += `result: ${JSON.stringify(it.result, null, 2)}\n\n`;
  }
  tools.scrollTop = tools.scrollHeight;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  sendBtn.disabled = true;

  addMsg('user', text);
  addMsg('assistant', '...');

  const placeholder = chat.lastChild;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const data = await res.json();
    placeholder.textContent = data.answer || '(empty)';
    setToolsLog(data.tool_log || []);
  } catch (e) {
    placeholder.textContent = 'ERROR: ' + e;
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) send();
});
