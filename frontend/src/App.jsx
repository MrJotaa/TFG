import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import AppLayout from './components/layout/AppLayout'
import ProtectedRoute from './components/ProtectedRoute'
import { AuthProvider } from './context/AuthContext'
import { ExperimentProvider } from './context/ExperimentContext'
import LoginPage from './pages/LoginPage'
import SignupPage from './pages/SignupPage'
import Dashboard from './pages/Dashboard'
import DatasetManager from './pages/DatasetManager'
import PredictionView from './pages/PredictionView'
import ComparisonView from './pages/ComparisonView'
import EvaluationView from './pages/EvaluationView'
import ProfilePage from './pages/ProfilePage'

function App() {
  return (
    <AuthProvider>
      <ExperimentProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />

            <Route element={<ProtectedRoute />}>
              <Route element={<AppLayout />}>
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/profile" element={<ProfilePage />} />
                <Route path="/datasets" element={<DatasetManager />} />
                <Route path="/prediction" element={<PredictionView />} />
                <Route path="/comparison" element={<ComparisonView />} />
                <Route path="/evaluation" element={<EvaluationView />} />
              </Route>
            </Route>

            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </ExperimentProvider>
    </AuthProvider>
  )
}

export default App