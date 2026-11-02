param(
    [Parameter(Mandatory=$true)][string]$Pdf,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$Width=2400
)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null=[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null=[Windows.Data.Pdf.PdfDocument,Windows.Data.Pdf,ContentType=WindowsRuntime]
$null=[Windows.Storage.Streams.InMemoryRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
$genericTask=[System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1
} | Select-Object -First 1
$actionTask=[System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and -not $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction'
} | Select-Object -First 1
function AwaitResult($Operation,[Type]$ResultType) {
    $task=$genericTask.MakeGenericMethod($ResultType).Invoke($null,@($Operation))
    $task.Wait()
    return $task.Result
}
$inputPath=(Resolve-Path -LiteralPath $Pdf).Path
$outputPath=[IO.Path]::GetFullPath($Output)
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($outputPath)) | Out-Null
$file=AwaitResult ([Windows.Storage.StorageFile]::GetFileFromPathAsync($inputPath)) ([Windows.Storage.StorageFile])
$document=AwaitResult ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
if ($document.PageCount -ne 1) { throw "Expected one-page figure, found $($document.PageCount)" }
$page=$document.GetPage(0)
$stream=[Windows.Storage.Streams.InMemoryRandomAccessStream]::new()
$options=[Windows.Data.Pdf.PdfPageRenderOptions]::new()
$options.DestinationWidth=$Width
$renderTask=$actionTask.Invoke($null,@($page.RenderToStreamAsync($stream,$options)))
$renderTask.Wait()
$reader=[System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream)
$destination=[IO.File]::Open($outputPath,[IO.FileMode]::Create)
try { $reader.CopyTo($destination) }
finally { $destination.Dispose(); $reader.Dispose(); $stream.Dispose(); $page.Dispose() }
Write-Output "Rendered one-page PDF: $outputPath"
