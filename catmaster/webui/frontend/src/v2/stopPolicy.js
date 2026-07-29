export function isEmergencyStopAttempt(attempt) {
  return Number(attempt) >= 3;
}
