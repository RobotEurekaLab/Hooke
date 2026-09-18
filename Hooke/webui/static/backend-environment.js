/* Read-only server checks shared by experiment pages. Configuration stays in the service account. */
async function refreshBackendEnvironment() {
  const status = document.getElementById('backend-environment');
  const details = document.getElementById('backend-environment-details');
  const report = document.getElementById('backend-environment-report');
  try {
    const response = await fetch('/api/backends/environment', {cache: 'no-store'});
    if (!response.ok) throw Error('Server environment check failed');
    const {isaac} = await response.json();
    status.textContent = isaac.installation_ready
      ? `Isaac installation accessible · GPU ${isaac.gpu} · Runtime verified when a scene opens`
      : `Isaac unavailable: ${isaac.error}`;
    report.textContent = [
      `Linux service account: ${isaac.service_user}`,
      `Installation: ${isaac.isaac_path || 'No accessible installation selected'}`,
      `Configuration source: ${isaac.configuration_source || 'Unavailable'}`,
      `Account configuration file: ${isaac.account_configuration}`,
      'Web visitors share this server configuration.',
      ...(isaac.discovery?.available_installations?.length
        ? ['Accessible installations discovered:', ...isaac.discovery.available_installations] : []),
      ...(isaac.blocked_path ? [`Blocked path: ${isaac.blocked_path}`] : []),
      ...isaac.remediation,
    ].join('\n');
    details.open = !isaac.installation_ready;
  } catch (error) {
    status.textContent = error.message;
  }
}
document.getElementById('backend-environment-refresh').onclick = refreshBackendEnvironment;
refreshBackendEnvironment();
