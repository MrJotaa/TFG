import { useEffect } from 'react'
import { getUsers } from './api/users'
import { getDatasets } from './api/datasets'
import { getModels, getActiveModel } from './api/models'

function App() {
  useEffect(() => {
    getUsers().then(console.log).catch(console.error)
    getDatasets().then(console.log).catch(console.error)
    getModels().then(console.log).catch(console.error)
    getActiveModel().then(console.log).catch(console.error)
  }, [])

  return (
    <div>
      <h1>Adversarial AI App</h1>
    </div>
  )
}

export default App