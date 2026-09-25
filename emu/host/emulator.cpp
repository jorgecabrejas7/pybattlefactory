// Headless mGBA host for Python (module `pybattle.pybattle_emu`).
//
// Runs the real ROM with no frontend: uncapped speed, in-memory savestates, direct bus
// reads/writes. Used as the exactness oracle and as a live-play backend.

#include <mgba/core/core.h>
#include <mgba/core/config.h>
#include <mgba/core/log.h>
#include <mgba-util/vfs.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void silentLog(struct mLogger*, int, enum mLogLevel, const char*, va_list) {}
struct mLogger gSilentLogger = {silentLog, nullptr};

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

class Emulator {
public:
    Emulator(const std::string& romPath, const std::string& savePath) {
        mLogSetDefaultLogger(&gSilentLogger);
        m_core = mCoreFind(romPath.c_str());
        if (!m_core) throw std::runtime_error("no mGBA core for " + romPath);
        if (!m_core->init(m_core)) throw std::runtime_error("mGBA core init failed");
        mCoreInitConfig(m_core, nullptr);

        unsigned w, h;
        m_core->desiredVideoDimensions(m_core, &w, &h);
        m_width = w;
        m_height = h;
        m_video.resize(static_cast<size_t>(w) * h);
        m_core->setVideoBuffer(m_core, m_video.data(), w);
        m_core->setAudioBufferSize(m_core, 2048);

        if (!mCoreLoadFile(m_core, romPath.c_str())) throw std::runtime_error("cannot load ROM " + romPath);
        if (!savePath.empty()) {
            // Keep save writes in memory so the user's .sav is never modified.
            m_save = readFile(savePath);
            struct VFile* vf = VFileMemChunk(m_save.data(), m_save.size());
            if (!m_core->loadSave(m_core, vf)) throw std::runtime_error("cannot load save " + savePath);
        }
        m_core->reset(m_core);
    }

    ~Emulator() {
        if (m_core) {
            mCoreConfigDeinit(&m_core->config);
            m_core->deinit(m_core);
        }
    }

    Emulator(const Emulator&) = delete;
    Emulator& operator=(const Emulator&) = delete;

    void reset() { m_core->reset(m_core); }
    void setKeys(uint32_t keys) { m_core->setKeys(m_core, keys); }
    uint32_t frame() const { return m_core->frameCounter(m_core); }

    void runFrames(int n) {
        py::gil_scoped_release release;
        for (int i = 0; i < n; i++) m_core->runFrame(m_core);
    }

    uint32_t read8(uint32_t a) { return m_core->busRead8(m_core, a); }
    uint32_t read16(uint32_t a) { return m_core->busRead16(m_core, a); }
    uint32_t read32(uint32_t a) { return m_core->busRead32(m_core, a); }

    py::bytes read(uint32_t addr, size_t size) {
        std::string out(size, '\0');
        for (size_t i = 0; i < size; i++) out[i] = static_cast<char>(m_core->busRead8(m_core, addr + i));
        return py::bytes(out);
    }

    void write(uint32_t addr, const std::string& data) {
        for (size_t i = 0; i < data.size(); i++)
            m_core->busWrite8(m_core, addr + i, static_cast<uint8_t>(data[i]));
    }

    void write32(uint32_t addr, uint32_t value) { m_core->busWrite32(m_core, addr, value); }

    py::bytes saveState() {
        std::string buf(m_core->stateSize(m_core), '\0');
        if (!m_core->saveState(m_core, buf.data())) throw std::runtime_error("saveState failed");
        return py::bytes(buf);
    }

    void loadState(const std::string& state) {
        if (state.size() != m_core->stateSize(m_core)) throw std::runtime_error("state size mismatch");
        if (!m_core->loadState(m_core, state.data())) throw std::runtime_error("loadState failed");
    }

    // Run frames until the u32 at `addr` (masked) equals `value`, checking after every frame.
    // Returns frames run, or -1 if `maxFrames` elapsed first.
    int runUntil32(uint32_t addr, uint32_t value, uint32_t mask, int maxFrames) {
        py::gil_scoped_release release;
        for (int i = 0; i <= maxFrames; i++) {
            if ((m_core->busRead32(m_core, addr) & mask) == value) return i;
            if (i < maxFrames) m_core->runFrame(m_core);
        }
        return -1;
    }

    // Run frames until any of the u32 at `addr` differs from its value at call time.
    int runUntilChange32(uint32_t addr, int maxFrames) {
        py::gil_scoped_release release;
        uint32_t start = m_core->busRead32(m_core, addr);
        for (int i = 1; i <= maxFrames; i++) {
            m_core->runFrame(m_core);
            if (m_core->busRead32(m_core, addr) != start) return i;
        }
        return -1;
    }

    py::array_t<uint8_t> screenshot() {
        py::array_t<uint8_t> img({static_cast<py::ssize_t>(m_height), static_cast<py::ssize_t>(m_width),
                                  static_cast<py::ssize_t>(3)});
        auto px = img.mutable_unchecked<3>();
        for (unsigned y = 0; y < m_height; y++) {
            for (unsigned x = 0; x < m_width; x++) {
                // mGBA's 32-bit color_t is XBGR8888
                uint32_t c = m_video[y * m_width + x];
                px(y, x, 0) = c & 0xFF;
                px(y, x, 1) = (c >> 8) & 0xFF;
                px(y, x, 2) = (c >> 16) & 0xFF;
            }
        }
        return img;
    }

private:
    struct mCore* m_core = nullptr;
    std::vector<color_t> m_video;
    std::vector<uint8_t> m_save;
    unsigned m_width = 0, m_height = 0;
};

}  // namespace

PYBIND11_MODULE(pybattle_emu, m) {
    m.doc() = "Headless mGBA host";

    // GBA key bits (mGBA GBA_KEY_*)
    m.attr("KEY_A") = 1 << 0;
    m.attr("KEY_B") = 1 << 1;
    m.attr("KEY_SELECT") = 1 << 2;
    m.attr("KEY_START") = 1 << 3;
    m.attr("KEY_RIGHT") = 1 << 4;
    m.attr("KEY_LEFT") = 1 << 5;
    m.attr("KEY_UP") = 1 << 6;
    m.attr("KEY_DOWN") = 1 << 7;
    m.attr("KEY_R") = 1 << 8;
    m.attr("KEY_L") = 1 << 9;

    py::class_<Emulator>(m, "Emulator")
        .def(py::init<const std::string&, const std::string&>(), py::arg("rom"), py::arg("save") = "")
        .def("reset", &Emulator::reset)
        .def("set_keys", &Emulator::setKeys)
        .def_property_readonly("frame", &Emulator::frame)
        .def("run_frames", &Emulator::runFrames, py::arg("n") = 1)
        .def("read8", &Emulator::read8)
        .def("read16", &Emulator::read16)
        .def("read32", &Emulator::read32)
        .def("read", &Emulator::read)
        .def("write", &Emulator::write)
        .def("write32", &Emulator::write32)
        .def("save_state", &Emulator::saveState)
        .def("load_state", &Emulator::loadState)
        .def("run_until32", &Emulator::runUntil32, py::arg("addr"), py::arg("value"),
             py::arg("mask") = 0xFFFFFFFFu, py::arg("max_frames") = 600)
        .def("run_until_change32", &Emulator::runUntilChange32, py::arg("addr"), py::arg("max_frames") = 600)
        .def("screenshot", &Emulator::screenshot);
}
