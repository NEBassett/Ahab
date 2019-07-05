#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
#include <vector>
#include <complex>
#include <random>
#include <math.h>
#include <utility>
#include <algorithm>
#include <boost/hana/tuple.hpp>
#include <boost/hana/zip.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/tuple.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "program.hpp"

// algorithm overview:
// float f_i, p, feq_i, u; f_i probability of streaming in i'th direction, p,
// macroscopic density at cell point, u macroscopic velocity at cell point, feq_i, equilibrium function for i'th direction at cell point
// core loop:
// 1 "stream" outward by setting f*_i(x, t) (* denoting that this is our intermediate, new f_i) = f_i(x - e_i*dx,t)
// 2 calculate new f_i with  f*_i and collision
// thus as f_i(x+ce_i*dt,t+dt) = f_i(x,t) - collision(f_i), thus f_i((x - e_i*dx) + ce_i*dt, t+dt) = f_i(x, t+dt) (as c = dx/dt)
// = f_i(x - e_i*dx, t) - collision(f_i(c - e_i*dx, t)) = f*_i - collision(f*_i)
// collision(f_i) = (f_i(x,t) - feq_i(x,t))/tau
// feq_i(x,t) = w_i * p + p*s_i(u(x,t) + tau*F/p)
// where tau is relaxation time

const std::string g_infoPath   = "../src/config.xml";
const std::string g_vertexPath = "../src/vertex.vs";
const std::string g_fragmentPath = "../src/fragment.fs";

void GLAPIENTRY msgCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam )
{
  std::cout << message << '\n';
}



struct simInfo
{
  size_t size;
  float dx, relaxationTime;

  simInfo(const std::string &filename)
  {
    boost::property_tree::ptree settings;
    boost::property_tree::xml_parser::read_xml(filename, settings);
    size = settings.get<unsigned int>("simInfo.size");
    dx = settings.get<float>("simInfo.dx");
    relaxationTime = settings.get<float>("simInfo.relaxationTime");
  }
};



struct simulationSpace
{
  GLuint vao;
  GLuint vbo;

  simulationSpace(const simInfo& info)
  {
    std::vector<glm::vec4> grid;

    for(auto i = size_t{0}; i < info.size; i++)
    {
      for(auto j = size_t{0}; j < info.size; j++)
      {
        grid.emplace_back(
          j,
          i,
          1.0f,
          1.0f
        );
      }
    }

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4)*grid.size(), grid.data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &vao);

    glBindVertexArray(vao);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void* )0);
    glEnableVertexAttribArray(0);
  }

  simulationSpace(const simulationSpace &) = delete; // don't copy this, handles imply owning semantics

  ~simulationSpace()
  {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
  }
};



struct screenQuad
{
  GLuint vao, vbo;

  screenQuad()
  {
    static auto verts = std::array<glm::vec4, 6>{
            glm::vec4(1.0f,  1.0f, 1.0f, 1.0f),
            glm::vec4(1.0f, -1.0f, 1.0f, 0.0f),
            glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f),
            glm::vec4(-1.0f, -1.0f, 0.0f, 0.0f),
            glm::vec4(1.0f, -1.0f, 1.0f, 0.0f),
            glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f)
    };

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4)*6, verts.data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
    glEnableVertexAttribArray(0);
  }

  screenQuad(const screenQuad &) = delete;

  ~screenQuad()
  {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
  }
};

class lbmsolver
{
  using streamType = decltype(
    GLDSEL::make_program_from_paths(
      boost::hana::make_tuple("", ""),
      glDselUniform("dim", int)
    )
  );

  using macType = decltype(
    GLDSEL::make_program_from_paths(
      boost::hana::make_tuple("", ""),
      glDselUniform("c", float),
      glDselUniform("tau", float),
      glDselUniform("time", float)
    )
  );

  using collType = decltype(
    GLDSEL::make_program_from_paths(
      boost::hana::make_tuple("", ""),
      glDselUniform("c", float),
      glDselUniform("tau", float)
    )
  );

  using fmapType = decltype(
    GLDSEL::make_program_from_paths(
      boost::hana::make_tuple("", "")
    )
  );

  GLuint grid_;
  GLuint boundary_;
  GLuint copy_;
  GLuint macvel_;
  GLuint macden_;
  GLuint force_;
  streamType stream_;
  macType calcMac_;
  collType collide_;
  fmapType foreach_;
  simInfo info_;
  std::array<float,9> default_dist_;

public:
  ~lbmsolver()
  {
    glDeleteTextures(1, &grid_);
    glDeleteTextures(1, &copy_);
    glDeleteTextures(1, &boundary_);
    glDeleteTextures(1, &macden_);
    glDeleteTextures(1, &macvel_);
  }

  lbmsolver(const lbmsolver &) = delete;

  auto bindResources()
  {
    glBindTexture(GL_TEXTURE_2D, macden_);
  }

  auto step(float tdelta, float time)
  {
    auto latticeSpeed = info_.dx/tdelta;

    glBindTexture(GL_TEXTURE_2D, boundary_);
    glBindImageTexture(0, grid_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, copy_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    stream_.setUniforms(
      glDselArgument("dim", info_.size)
    );

    glDispatchCompute(info_.size, info_.size, 1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    std::swap(grid_, copy_);

    glBindImageTexture(0, grid_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, macvel_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG32F);
    glBindImageTexture(2, macden_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    calcMac_.setUniforms(
      glDselArgument("c", latticeSpeed),
      glDselArgument("tau", info_.relaxationTime),
      glDselArgument("time", time)
    );

    glDispatchCompute(info_.size, info_.size, 1);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    glBindImageTexture(0, grid_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(1, macvel_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(2, macden_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(3, copy_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    collide_.setUniforms(
      glDselArgument("c", latticeSpeed),
      glDselArgument("tau", info_.relaxationTime)
    );

    glDispatchCompute(info_.size, info_.size*9, 1);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    std::swap(grid_, copy_);
  }

  lbmsolver(const simInfo &info) :
    grid_(0),
    boundary_(0),
    copy_(0),
    macvel_(0),
    macden_(0),
    stream_(
      GLDSEL::make_program_from_paths(
      boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/stream/main.cs"),
      glDselUniform("dim", int)
      )
    ),
    calcMac_(
      GLDSEL::make_program_from_paths(
      boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/macrocalc/main.cs"),
      glDselUniform("c", float),
      glDselUniform("tau", float),
      glDselUniform("time", float)
      )
    ),
    collide_(
      GLDSEL::make_program_from_paths(
      boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/collision/main.cs"),
      glDselUniform("c", float),
      glDselUniform("tau", float)
      )
    ),
    foreach_(
      GLDSEL::make_program_from_paths(
      boost::hana::make_tuple(boost::none,boost::none,boost::none,boost::none,boost::none, "../src/foreach/main.cs")
      )
    ),
    info_(info),
    default_dist_({1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f,1.0f/9.0f,4.0f/9.0f, 1.0f/9.0f, 1.0f/36.0f, 1.0f/36.0f,1.0f/9.0f})
  {
    const auto newTex = [this](auto&& seed, GLenum border = GL_CLAMP_TO_EDGE)
    {

      GLuint newV;
      glGenTextures(1, &newV);
      glBindTexture(GL_TEXTURE_2D, newV);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, info_.size, info_.size*9, 0, GL_RED, GL_FLOAT, nullptr);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, border);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, border);

      glBindImageTexture(0, newV, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

      foreach_.activate();

      glDispatchCompute(info_.size, info_.size*9, 1);

      return newV;
    };

    copy_ = newTex([](auto x, auto y, auto i){ return float{0}; });
    grid_ = newTex([this](auto x, auto y, auto z){ return x/(9.0f); });

    std::vector<float> data(info_.size*info_.size, 0);

    glGenTextures(1, &boundary_);
    glBindTexture(GL_TEXTURE_2D, boundary_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, info_.size, info_.size, 0, GL_RED, GL_FLOAT, data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    float color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, color);

    glGenTextures(1, &macvel_);
    glBindTexture(GL_TEXTURE_2D, macvel_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, info_.size, info_.size, 0, GL_RG, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &macden_);
    glBindTexture(GL_TEXTURE_2D, macden_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, info_.size, info_.size, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &force_);
    glBindTexture(GL_TEXTURE_2D, force_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, info_.size, info_.size, 0, GL_RG, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
};



int main()
{
  try 
  {
     glfwSetErrorCallback([](auto err, const auto* desc){ std::cout << "Error: " << desc << '\n'; });

  // glfw init
  if(!glfwInit())
  {
    throw std::runtime_error(
        "GLFW failed to initialize"
    );
  }

  // context init
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  auto window = glfwCreateWindow(640, 480, "Lattice Boltzmann Methods", NULL, NULL);
  if (!window)
  {
    throw std::runtime_error(
        "GLFW window creation failed"
    );
  }

  glfwMakeContextCurrent(window);

  // glew init
  auto err = glewInit();
  if(GLEW_OK != err)
  {
    throw std::runtime_error(
        std::string("GLEW initialization failed with error: ") + std::string((const char*)(glewGetErrorString(err)))
    );
  }

  // gl init
  glEnable(GL_DEBUG_OUTPUT);
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glPointSize(15.5f);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glDebugMessageCallback(msgCallback, 0);
  glEnable(GL_DEPTH_TEST);



  // program initialization
  auto siminfo = simInfo{g_infoPath};

  screenQuad quad{};
  lbmsolver fsim{siminfo};
  simulationSpace space{siminfo};

  auto gridDraw = GLDSEL::make_program_from_paths(
      boost::hana::make_tuple("../src/main/vertex.vs", "../src/main/fragment.fs"),
      glDselUniform("time", float),
      glDselUniform("model", glm::mat4),
      glDselUniform("view", glm::mat4),
      glDselUniform("proj", glm::mat4)
  );

  int width, height;
  double time = glfwGetTime();
  glm::mat4 proj;
  double tdelta;
  double temp;

  glfwSetWindowUserPointer(window, &fsim);

   glfwSetKeyCallback(window, [](auto *window, auto key, auto, auto action, auto mods){
     auto fsimptr = static_cast<lbmsolver*>(glfwGetWindowUserPointer(window));

     if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
       glfwSetWindowShouldClose(window, GLFW_TRUE);
     if(key == GLFW_KEY_F4)
     { /* later */ }
   });



  glfwSwapInterval(1);em
  {
    auto oldT = time;
    time = glfwGetTime();

    fsim.step(1.0f/60.0f, time);

    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    proj = glm::perspective(1.57f, float(width)/float(height), 0.1f, 7000.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    fsim.bindResources();

    //draw(proj, view, glm::mat4(), glfwGetTime(), plane);
    gridDraw.setUniforms( // set uniforms
      glDselArgument("model", glm::mat4()),
      glDselArgument("view", glm::mat4()),
      glDselArgument("proj", proj),
      glDselArgument("time", float(glfwGetTime()))
    );

    glBindVertexArray(quad.vao);

    glDrawArrays(GL_TRIANGLES, 0, 6); // draw quad

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate(); 
  } catch(const std::exception& e){
    std::cout << "main failed with " << e.what() << '\n';
  }
  
  return 0;
}
