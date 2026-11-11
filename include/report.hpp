#pragma once

#include <string>
#include <vector>
#include "optimization.hpp"

// ---------------------------------------------------------------------------
// report — console and CSV output for optimizer experiments.
//
// The CSVs open directly in Excel / pandas, so experiments can be compared
// outside the application.
// ---------------------------------------------------------------------------
namespace report {

/// Print one block per optimizer to stdout (losses, updates, time, convergence).
void print(const std::vector<optim::OptimizerResult>& results);

/// One row per optimizer:
///   optimizer,initial_loss,final_loss,epochs_to_converge,updates,time_ms,converged
void save_summary_csv(const std::vector<optim::OptimizerResult>& results,
                      const std::string& filename);

/// One row per (optimizer, epoch) — "long" format, plots directly as one line
/// per optimizer:
///   optimizer,epoch,loss
void save_loss_history_csv(const std::vector<optim::OptimizerResult>& results,
                           const std::string& filename);

} // namespace report
