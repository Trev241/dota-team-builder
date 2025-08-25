import { createRouter, createWebHistory } from "vue-router"
import Team from "../views/Team.vue"
import About from "../views/About.vue"

const routes = [
  { path: "/", name: "Team", component: Team },
  { path: "/about", name: "About", component: About },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
