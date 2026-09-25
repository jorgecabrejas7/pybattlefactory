--[[
pybattle bridge for mGBA (0.10+) — lets pybattle drive the game running in mgba-qt so you can
watch agents play live.

  1. Open the ROM (and your save) in mgba-qt
  2. Tools -> Scripting -> File -> Load script -> this file
  3. In Python:  EmuBackend(machine=LuaMachine())   (see pybattle/emu/lua_machine.py)

Protocol: one command per line, one reply line per command.
  R <addr> <len>     read bytes             -> hex
  W <addr> <hex>     write bytes            -> OK
  K <mask>           hold these keys        -> OK
  F <n>              run n frames           -> OK (sent after n frames have elapsed)
  N                  current frame number   -> number
  S                  save state             -> hex
  L <hex>            load state             -> OK
--]]

local PORT = 8765
local server = assert(socket.bind(nil, PORT), "pybattle bridge: cannot bind port " .. PORT)
assert(server:listen())
console:log("pybattle bridge listening on port " .. PORT)

local client = nil
local buffer = ""
local frame = 0
local waitUntil = nil          -- frame number that completes a pending F command

local function reply(s)
    if client then client:send(s .. "\n") end
end

local function tohex(s)
    return (s:gsub(".", function(c) return string.format("%02x", string.byte(c)) end))
end

local function fromhex(h)
    return (h:gsub("..", function(cc) return string.char(tonumber(cc, 16)) end))
end

local function handle(line)
    local cmd, a, b = line:match("^(%S+)%s*(%S*)%s*(%S*)")
    if cmd == "R" then
        reply(tohex(emu:readRange(tonumber(a), tonumber(b))))
    elseif cmd == "W" then
        local addr, data = tonumber(a), fromhex(b)
        for i = 1, #data do emu:write8(addr + i - 1, string.byte(data, i)) end
        reply("OK")
    elseif cmd == "K" then
        emu:setKeys(tonumber(a))
        reply("OK")
    elseif cmd == "F" then
        waitUntil = frame + tonumber(a)
    elseif cmd == "N" then
        reply(tostring(frame))
    elseif cmd == "S" then
        reply(tohex(emu:saveStateBuffer()))
    elseif cmd == "L" then
        emu:loadStateBuffer(fromhex(a))
        reply("OK")
    else
        reply("ERR unknown command")
    end
end

local function pump()
    if waitUntil then return end           -- an F command is still running
    while true do
        local nl = buffer:find("\n")
        if not nl then return end
        local line = buffer:sub(1, nl - 1)
        buffer = buffer:sub(nl + 1)
        handle(line)
        if waitUntil then return end
    end
end

server:add("received", function()
    if client then return end
    client = server:accept()
    console:log("pybattle bridge: client connected")
    client:add("received", function()
        local data = client:receive(65536)
        if data then buffer = buffer .. data end
        pump()
    end)
    client:add("error", function()
        console:log("pybattle bridge: client disconnected")
        client, buffer, waitUntil = nil, "", nil
    end)
end)

callbacks:add("frame", function()
    frame = frame + 1
    if waitUntil and frame >= waitUntil then
        waitUntil = nil
        reply("OK")
        pump()
    end
    server:poll()
    if client then client:poll() end
end)
