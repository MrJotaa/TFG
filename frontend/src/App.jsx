import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import AppLayout from './components/layout/AppLayout'
import ProtectedRoute from './components/ProtectedRoute'
import { AuthProvider } from './context/AuthContext'
import { ExperimentProvider } from './context/ExperimentContext'
import { NavigationGuardProvider } from './context/NavigationGuardContext'
import LoginPage from './pages/LoginPage'
import SignupPage from './pages/SignupPage'
import Dashboard from './pages/Dashboard'
import DatasetManager from './pages/DatasetManager'
import ComparisonView from './pages/ComparisonView'
import EvaluationComparisonView from './pages/EvaluationComparisonView'
import EvaluationView from './pages/EvaluationView'
import IndividualAnalysisComparisonView from './pages/IndividualAnalysisComparisonView'
import ProfilePage from './pages/ProfilePage'

function App() {
  return (
    <AuthProvider>
      <ExperimentProvider>
        <BrowserRouter>
          <NavigationGuardProvider>
            <Routes>
            <Route path="/" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />

            <Route element={<ProtectedRoute />}>
              <Route element={<AppLayout />}>
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/profile" element={<ProfilePage />} />
                <Route path="/datasets" element={<DatasetManager />} />
                <Route
                  path="/prediction"
                  element={<ComparisonView key="prediction" mode="prediction" />}
                />
                <Route
                  path="/detection"
                  element={<Navigate to="/prediction" replace />}
                />
                <Route path="/comparison" element={<ComparisonView key="attack" mode="attack" />} />
                <Route path="/evaluation" element={<EvaluationView />} />
                <Route
                  path="/evaluations/compare"
                  element={<EvaluationComparisonView />}
                />
                <Route
                  path="/individual-analyses/compare"
                  element={<IndividualAnalysisComparisonView />}
                />
              </Route>
            </Route>

            <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </NavigationGuardProvider>
        </BrowserRouter>
      </ExperimentProvider>
    </AuthProvider>
  )
}

export default App
