#include "gen3_game.hpp"

extern "C" {
#include "gen3/host.h"
#include "gen3/search_host.h"
}

#include <stdexcept>

namespace pkmn {

namespace {

// The instance whose state currently occupies the live battle RAM section.
Gen3Game* s_live = nullptr;

const std::vector<uint8_t>& pristineState() {
    static std::vector<uint8_t> state = [] {
        Gen3_InitSystem();
        std::vector<uint8_t> s(Gen3_StateSize());
        Gen3_SaveState(s.data());
        return s;
    }();
    return state;
}

}  // namespace

Gen3Game::Gen3Game() : m_state(pristineState()) {}

Gen3Game::Gen3Game(const Gen3Game& other) {
    if (s_live == &other) Gen3_SaveState(const_cast<uint8_t*>(other.m_state.data()));
    m_state = other.m_state;
}

Gen3Game& Gen3Game::operator=(const Gen3Game& other) {
    if (this != &other) {
        if (s_live == &other) Gen3_SaveState(const_cast<uint8_t*>(other.m_state.data()));
        m_state = other.m_state;
        if (s_live == this) Gen3_LoadState(m_state.data());
    }
    return *this;
}

Gen3Game::~Gen3Game() {
    if (s_live == this) s_live = nullptr;
}

void Gen3Game::activate() {
    if (s_live == this) return;
    if (s_live) Gen3_SaveState(s_live->m_state.data());
    Gen3_LoadState(m_state.data());
    s_live = this;
}

void Gen3Game::setBattle(uint32_t flags, uint16_t trainerId) {
    activate();
    Gen3_SetBattleParams(flags, trainerId);
}

void Gen3Game::writeParty(int side, const std::string& raw) {
    activate();
    Gen3_WriteParty(side, raw.data(), raw.size());
}

void Gen3Game::writeSaveBlock2(uint32_t offset, const std::string& data) {
    activate();
    Gen3_WriteSaveBlock2(offset, data.data(), data.size());
}

std::string Gen3Game::readSaveBlock2(uint32_t offset, size_t size) {
    activate();
    std::string out(size, '\0');
    Gen3_ReadSaveBlock2(offset, out.data(), size);
    return out;
}

void Gen3Game::setFlag(uint16_t id, bool on) {
    activate();
    Gen3_SetFlag(id, on ? 1 : 0);
}

void Gen3Game::setVar(uint16_t id, uint16_t value) {
    activate();
    Gen3_SetVar(id, value);
}

bool Gen3Game::start(uint32_t rng) {
    activate();
    if (!Gen3_StartBattle(10000)) return false;
    Gen3_SetRng(rng);
    return true;
}

int Gen3Game::run(uint32_t maxFrames) {
    activate();
    return Gen3_RunUntilDecision(maxFrames);
}

void Gen3Game::chooseMove(uint8_t slot) {
    activate();
    if (Gen3_Host()->pendingDecision != ACTION) throw std::runtime_error("not at an action decision");
    Gen3_ChooseMove(slot);
}

void Gen3Game::chooseSwitch(uint8_t partyIndex) {
    activate();
    int pending = Gen3_Host()->pendingDecision;
    if (pending != ACTION && pending != SWITCH) throw std::runtime_error("not at a decision");
    if (pending == ACTION && !Gen3_CanSwitch(0)) throw std::runtime_error("cannot switch: trapped");
    Gen3_ChooseSwitch(partyIndex);
}

uint8_t Gen3Game::unusableMoves(uint8_t battler) {
    activate();
    return Gen3_UnusableMoves(battler);
}

void Gen3Game::forfeit() {
    activate();
    if (Gen3_Host()->pendingDecision != ACTION) throw std::runtime_error("not at an action decision");
    Gen3_Forfeit();
}

bool Gen3Game::canSwitch(uint8_t battler) {
    activate();
    return Gen3_CanSwitch(battler) != 0;
}

uint16_t Gen3Game::choicedMove(uint8_t battler) {
    activate();
    return Gen3_ChoicedMove(battler);
}

uint32_t Gen3Game::rng() {
    activate();
    return Gen3_GetRng();
}

std::string Gen3Game::readRam(uint32_t addr, size_t size) {
    activate();
    std::string out(size, '\0');
    if (!Gen3_ReadRam(addr, out.data(), size)) throw std::out_of_range("address not in host battle RAM");
    return out;
}

void Gen3Game::factoryBegin(bool openLevel, uint16_t winStreak, uint16_t rentsCount, uint32_t seed) {
    activate();
    Gen3Factory_Begin(openLevel ? 1 : 0, winStreak, rentsCount, seed);
}

int Gen3Game::factoryPhase() {
    activate();
    return Gen3Factory_Phase();
}

std::string Gen3Game::factoryRental(uint8_t i) {
    activate();
    std::string out(100, '\0');
    Gen3Factory_ReadRental(i, out.data());
    return out;
}

uint16_t Gen3Game::factoryRentalMonId(uint8_t i) {
    activate();
    return Gen3Factory_RentalFrontierMonId(i);
}

bool Gen3Game::factoryRent(uint8_t a, uint8_t b, uint8_t c) {
    activate();
    return Gen3Factory_Rent(a, b, c) != 0;
}

bool Gen3Game::factorySwap(int playerSlot, int enemySlot) {
    activate();
    return Gen3Factory_Swap(playerSlot, enemySlot) != 0;
}

int Gen3Game::factoryRunBattle(uint32_t maxFrames) {
    activate();
    return Gen3Factory_RunBattle(maxFrames);
}

Gen3Game::FactoryInfo Gen3Game::factoryInfo() {
    activate();
    const Gen3FactoryState* s = Gen3Factory_State();
    return {s->phase, s->lvlMode, s->hintType, s->hintStyle, s->brainStatus, s->lastOutcome,
            s->trainerId, s->wins, s->swaps, s->challengesWon};
}

void Gen3Game::writeRam(uint32_t addr, const std::string& data) {
    activate();
    if (!Gen3_WriteRam(addr, data.data(), data.size())) throw std::out_of_range("address not in host battle RAM");
}

uint32_t Gen3Game::frames() {
    activate();
    return Gen3_Host()->frames;
}

void Gen3Game::traceRandom(bool enable) {
    Gen3_TraceRandom(enable ? 1 : 0);
}

std::pair<int, int> Gen3Game::simStep(int kind, int index) {
    activate();
    if (Gen3_BattleOutcome() != 0) throw std::runtime_error("battle is over");
    if (kind == 0) {
        if (index < 0 || index > 3) throw std::out_of_range("move slot");
        chooseMove(static_cast<uint8_t>(index));
    } else if (kind == 1) {
        if (index < 0 || index > 5) throw std::out_of_range("party index");
        chooseSwitch(static_cast<uint8_t>(index));
    } else {
        throw std::invalid_argument("kind must be 0 (move) or 1 (switch)");
    }
    // Same frame budget as Gen3Factory_RunBattle (SimBackend), without its FinishBattle
    int decision = Gen3_RunUntilDecision(400000);
    return {decision, Gen3_BattleOutcome()};
}

void Gen3Game::setRng(uint32_t value) {
    activate();
    Gen3_SetRng(value);
}

int Gen3Game::battleOutcome() {
    activate();
    return Gen3_BattleOutcome();
}

void Gen3Game::determinize(const std::vector<DetSlot>& slots, int64_t hiddenSeed) {
    activate();
    std::vector<Gen3DetSlot> c;
    c.reserve(slots.size());
    for (const DetSlot& s : slots) c.push_back({s.partySlot, s.setId, s.iv, s.abilityBit, s.hpFraction});
    if (Gen3Search_Determinize(c.data(), static_cast<int>(c.size()), hiddenSeed) != 0)
        throw std::invalid_argument("bad determinization spec");
}

}  // namespace pkmn
