#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal>
#include <lgpio.h>
#include <string>

// On Raspberry Pi 5 / CM5, the primary 40-pin header is managed by the RP1 chip, 
// which is typically exposed as gpiochip4.
const int GPIO_CHIP = 4; 

// The single control pin connected to the MOSFET gate driving all 4 LEDs
const int CONTROL_PIN = 17;

// Global flag for clean exit on Ctrl+C
volatile sig_atomic_t keep_running = 1;

void sig_handler(int sig) {
    keep_running = 0;
}

// Generate the OOK packet array: [1 Start] + [0 Sync] + [10-bit ID] + [5 Rest]
std::vector<int> build_packet(uint16_t marker_id) {
    std::vector<int> packet;
    
    // 1. Start Bit (High) - Wakes up the receiver
    packet.push_back(1);
    
    // 2. Sync Bit (Low) - Creates a guaranteed falling edge
    packet.push_back(0);
    
    // 3. Payload: 10-bit ID (MSB first)
    for (int i = 9; i >= 0; --i) {
        packet.push_back((marker_id >> i) & 1);
    }
    
    // 4. Rest Period (Low) - 5 bit-times to separate transmissions
    for (int i = 0; i < 5; ++i) {
        packet.push_back(0);
    }
    
    return packet;
}

int main(int argc, char* argv[]) {
    double fps = 100.0; // default
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--fps" && i + 1 < argc) {
            fps = std::stod(argv[++i]);
        }
    }
    // 1/fps * 3 seconds = 3000/fps milliseconds
    auto bit_duration = std::chrono::milliseconds(static_cast<int>(3000.0 / fps));

    // Register signal handler for clean shutdown
    std::signal(SIGINT, sig_handler);

    // Open the GPIO chip
    int handle = lgGpiochipOpen(GPIO_CHIP);
    if (handle < 0) {
        std::cerr << "Failed to open gpiochip" << GPIO_CHIP << " (Error " << handle << ")\n";
        return 1;
    }

    // Claim the single control pin as output
    if (lgGpioClaimOutput(handle, 0, CONTROL_PIN, 0) < 0) {
        std::cerr << "Failed to claim GPIO " << CONTROL_PIN << "\n";
        lgGpiochipClose(handle);
        return 1;
    }

    // Set the Marker ID you want to broadcast (0 to 1023)
    uint16_t my_marker_id = 42; 
    std::vector<int> packet = build_packet(my_marker_id);

    std::cout << "Broadcasting Marker ID " << my_marker_id << " on Pin " << CONTROL_PIN << "...\n";
    std::cout << "Press Ctrl+C to stop.\n";

    // Setup absolute timing baseline
    auto next_tick = std::chrono::steady_clock::now();

    while (keep_running) {
        for (int bit : packet) {
            if (!keep_running) break;

            // Write the current bit to the single control pin
            lgGpioWrite(handle, CONTROL_PIN, bit);

            // Calculate the exact time the NEXT bit should occur
            next_tick += bit_duration;

            // Sleep exactly until that absolute time to prevent drift
            std::this_thread::sleep_until(next_tick);
        }
    }

    // Clean up: turn off the pin and release it
    std::cout << "\nShutting down...\n";
    lgGpioWrite(handle, CONTROL_PIN, 0);
    lgGpiochipClose(handle);

    return 0;
}