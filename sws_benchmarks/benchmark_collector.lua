-- Script that runs at wrk's "done" stage collecting the stats in JSON and CSV formats.
-- https://github.com/wg/wrk/blob/master/SCRIPTING

local server_name = os.getenv("SERVER")

done = function(summary, latency, requests)
    os.execute("mkdir -p data/")

    local filename = "benchmark_" .. server_name
    local file_path = "data/" .. "/" .. filename

    local csv = ''
    -- csv = csv .. string.format('requests,')
    -- csv = csv .. string.format('duration_ms,')
    -- csv = csv .. string.format('requests_per_sec,')
    -- csv = csv .. string.format('bytes,')
    -- csv = csv .. string.format('bytes_transfer_per_sec,')
    -- csv = csv .. string.format('connect_errors,')
    -- csv = csv .. string.format('read_errors,')
    -- csv = csv .. string.format('write_errors,')
    -- csv = csv .. string.format('http_errors,')
    -- csv = csv .. string.format('timeouts,')
    -- csv = csv .. string.format('latency_min,')
    -- csv = csv .. string.format('latency_max,')
    -- csv = csv .. string.format('latency_mean_ms,')
    -- csv = csv .. string.format('latency_stdev\n')

    csv = csv .. string.format('%d,', summary.requests)
    csv = csv .. string.format('%0.5f,', summary.duration / 1000)
    csv = csv .. string.format('%0.5f,', (summary.requests / summary.duration) * 1e6)
    csv = csv .. string.format('%d,', summary.bytes)
    csv = csv .. string.format('%0.5f,', (summary.bytes / summary.duration) * 1e6)
    csv = csv .. string.format('%d,', summary.errors.connect)
    csv = csv .. string.format('%d,', summary.errors.read)
    csv = csv .. string.format('%d,', summary.errors.write)
    csv = csv .. string.format('%d,', summary.errors.status)
    csv = csv .. string.format('%d,', summary.errors.timeout)
    csv = csv .. string.format('%0.2f,', latency.min)
    csv = csv .. string.format('%0.2f,', latency.max)
    csv = csv .. string.format('%0.2f,', latency.mean / 1000)
    csv = csv .. string.format('%0.2f\n', latency.stdev)

    local file, err = io.open(file_path .. ".csv", "a")
    if file then
        file:write(csv)
        file:close()
    else
        print("error saving csv results file:", err)
    end
end
