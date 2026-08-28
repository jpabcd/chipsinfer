const $ = (id) => document.getElementById(id);

function showSummary(summary) {
  const number = (value, suffix = '') => value == null ? '-' : `${value}${suffix}`;
  const values = [
    ['检测矩形', summary.rect_count],
    ['机械记录', summary.mechanical_count],
    ['已匹配', summary.matched_count],
    ['未匹配矩形', summary.unmatched_rect_count],
    ['未匹配机械记录', summary.unmatched_mechanical_count],
    ['RMSE', number(summary.rmse_um == null ? null : summary.rmse_um.toFixed(2), ' μm')],
    ['最大残差', number(summary.max_residual_um == null ? null : summary.max_residual_um.toFixed(2), ' μm')],
    ['X 反向', summary.x_reversed ? '是' : '否'],
    ['Y 反向', summary.y_reversed ? '是' : '否'],
  ];
  $('summary').innerHTML = values.map(([name, value]) => `<div><span>${name}</span><strong>${value}</strong></div>`).join('');
  $('summary').classList.remove('hidden');
}

function showRecords(records) {
  $('records').innerHTML = records.map((item) => {
    const b = item.box;
    const position = `${b.x}, ${b.y}, ${b.w}×${b.h}`;
    const status = item.matched ? '已匹配' : '未匹配';
    return `<tr class="${item.matched ? '' : 'unmatched'}"><td>${item.rect_id}</td><td>${position}</td><td>${item.mx ?? '-'}</td><td>${item.my ?? '-'}</td><td>${item.residual_um == null ? '-' : item.residual_um.toFixed(2)}</td><td>${status}</td></tr>`;
  }).join('');
}

$('run').addEventListener('click', async () => {
  const button = $('run');
  button.disabled = true;
  $('message').textContent = '正在读取 MSK、解析 TXT 并执行匹配……';
  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        config_path: $('configPath').value.trim(),
        msk_path: $('mskPath').value.trim(),
        txt_path: $('txtPath').value.trim(),
      }),
    });
    const result = await response.json();
    if (!response.ok || result.error) throw new Error(result.error || '匹配失败');
    $('msk').src = result.msk_image;
    $('overlay').src = result.overlay_image;
    $('txt').textContent = result.txt;
    showSummary(result.summary);
    showRecords(result.records);
    $('message').textContent = `${result.message || '匹配完成'}  文件：${result.msk_path}`;
  } catch (error) {
    $('message').textContent = `错误：${error.message}`;
  } finally {
    button.disabled = false;
  }
});
