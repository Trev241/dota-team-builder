import { createRouter, createWebHistory } from "vue-router"
import Team from "../views/Team.vue"
import About from "../views/About.vue"
import Embeddings from "../views/Synergies.vue"

const routes = [
  { path: "/", name: "Team", component: Team },
  { path: "/about", name: "About", component: About },
  {
    path: "/embeddings",
    name: "Hero Embeddings",
    component: Embeddings,
    meta: { hideNavbar: false },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) return savedPosition
    else if (to.hash) return { el: to.hash, behavior: "smooth" }
    else return { top: 0 }
  },
})

export default router
