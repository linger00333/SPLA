$ErrorActionPreference = 'Stop'
$taskPlan = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'api_publication_plan.json') -Raw | ConvertFrom-Json
$taskRepoPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\github\SPLA'))
$env:GIT_TERMINAL_PROMPT = '0'
$taskCredentialLines = @('protocol=https','host=github.com','') | git -C $taskRepoPath credential fill
if ($LASTEXITCODE -ne 0) { throw 'Existing GitHub identity unavailable' }
$taskCredential = @{}
foreach ($taskLine in $taskCredentialLines) {
    $taskSplit = $taskLine.IndexOf('=')
    if ($taskSplit -gt 0) { $taskCredential[$taskLine.Substring(0,$taskSplit)] = $taskLine.Substring($taskSplit+1) }
}
if (-not $taskCredential['password']) { throw 'No existing write credential' }
$taskHeaders = @{Authorization=('Bearer ' + $taskCredential['password']); Accept='application/vnd.github+json'; 'X-GitHub-Api-Version'='2022-11-28'}
$taskBaseUrl = 'https://api.github.com/repos/' + $taskPlan.repository
function Invoke-PublicationApi($method,$path,$body) {
    if ($null -eq $body) { return Invoke-RestMethod -Method $method -Uri ($taskBaseUrl+$path) -Headers $taskHeaders }
    $taskJson = $body | ConvertTo-Json -Depth 80 -Compress
    return Invoke-RestMethod -Method $method -Uri ($taskBaseUrl+$path) -Headers $taskHeaders -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($taskJson))
}
try {
    $taskHead = Invoke-PublicationApi GET '/git/ref/heads/main' $null
    if ($taskHead.object.sha -ne $taskPlan.base) { throw 'Remote branch moved; publication stopped before ref update' }
    $taskParent = $taskPlan.base
    $taskCreated = @()
    foreach ($taskCommit in $taskPlan.commits) {
        $taskParentInfo = Invoke-PublicationApi GET ('/git/commits/'+$taskParent) $null
        $taskEntries = @()
        foreach ($taskEntry in $taskCommit.entries) {
            if ($taskEntry.PSObject.Properties.Name -contains 'base64') {
                $taskBlob = Invoke-PublicationApi POST '/git/blobs' @{content=$taskEntry.base64;encoding='base64'}
                $taskEntries += @{path=$taskEntry.path;mode=$taskEntry.mode;type='blob';sha=$taskBlob.sha}
            } else {
                $taskEntries += $taskEntry
            }
        }
        $taskTree = Invoke-PublicationApi POST '/git/trees' @{base_tree=$taskParentInfo.tree.sha;tree=$taskEntries}
        if ($taskTree.sha -ne $taskCommit.tree) { throw 'Created tree differs from reviewed tree' }
        $taskBody = @{message=$taskCommit.commit.message;tree=$taskTree.sha;parents=@($taskParent);author=$taskCommit.commit.author;committer=$taskCommit.commit.committer}
        $taskObject = Invoke-PublicationApi POST '/git/commits' $taskBody
        $taskCreated += $taskObject
        $taskParent = $taskObject.sha
    }
    $taskCurrent = Invoke-PublicationApi GET '/git/ref/heads/main' $null
    if ($taskCurrent.object.sha -ne $taskPlan.base) { throw 'Remote branch changed; ref update cancelled' }
    $taskUpdated = Invoke-PublicationApi PATCH '/git/refs/heads/main' @{sha=$taskParent;force=$false}
    if ($taskUpdated.object.sha -ne $taskParent) { throw 'Final ref mismatch' }
    [pscustomobject]@{base=$taskPlan.base;local_previous_head=$taskPlan.local_head;published_head=$taskParent;tree=$taskPlan.expected_final_tree;commits=$taskCreated;force=$false} | ConvertTo-Json -Depth 80 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'api_publication_receipt.json') -Encoding utf8
    [pscustomobject]@{published_head=$taskParent;tree=$taskPlan.expected_final_tree;commits=$taskCreated.Count;force=$false} | ConvertTo-Json
} finally {
    $taskHeaders.Clear();$taskCredential.Clear();$taskCredentialLines=$null
}
