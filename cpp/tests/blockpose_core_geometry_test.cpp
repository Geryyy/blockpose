#include <blockpose/core/geometry.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace blockpose::core { namespace {

Points plane_points() {Points p; for (int x=-5;x<=5;++x) for (int y=-5;y<=5;++y) p.emplace_back(x*.1,y*.1,0.); p.emplace_back(3.,2.,1.); return p;}

TEST(Geometry, SamplerIsRepeatableAndBounded) {DeterministicSampler first(42U), second(42U); for (std::size_t i=0;i<64U;++i) {const auto a=first.index(121U), b=second.index(121U); EXPECT_EQ(a,b); EXPECT_LT(a,121U);}}

TEST(Geometry, RansacIsDeterministicAndRefitsFullConsensus) {const auto p=plane_points(); RansacParameters params; params.distance_threshold=.001; params.num_iterations=300; params.seed=42; const auto first=segment_plane(p,params); auto shuffled=p; std::reverse(shuffled.begin(),shuffled.end()); const auto second=segment_plane(shuffled,params); EXPECT_NEAR(std::abs(first.plane.normal.z()),1.,1e-12); EXPECT_NEAR(first.plane.offset,0.,1e-12); EXPECT_EQ(first.inlier_indices.size(),121U); EXPECT_EQ(first.inlier_indices,second.inlier_indices); EXPECT_EQ(first.diagnostics.trials_evaluated,second.diagnostics.trials_evaluated); EXPECT_GE(first.diagnostics.search_ms,0.0); EXPECT_GE(first.diagnostics.full_score_refit_ms,0.0);}

TEST(Geometry, GroundAndLocalSupportUsesNearbyCells) {GroundPlane plane{Point(0.,-.1,1.).normalized(),-.3,"ransac"}; Point surface(1.,.5,plane.support_z(Point(1.,.5,0.))); EXPECT_NEAR(plane.height(surface),0.,1e-12); LocalGroundModel local; local.normal=Point::UnitZ(); local.cell_size=.5; local.cell_z.emplace(std::make_pair(0LL,0LL),.1); EXPECT_NEAR(local.support_z(Point(.3,.2,5.)),.1,1e-12); EXPECT_NEAR(local.height(Point(.3,.2,.35)),.25,1e-12); EXPECT_NEAR(local.support_z(Point(3.,3.,5.)),0.,1e-12);}

TEST(Geometry, GroundRemovalUsesRansacAndPreservesObject) {Points p; for(int x=-5;x<=5;++x) for(int y=-5;y<=5;++y) p.emplace_back(x*.1,y*.1,.02*x*.1); p.emplace_back(.1,.1,.7); GroundParameters params; params.ransac_distance=.002; params.thickness=.15; params.local_clearance=.05; const auto result=remove_ground(p,params); EXPECT_EQ(result.ground.source,"ransac"); ASSERT_EQ(result.above_ground.size(),1U); EXPECT_NEAR(result.above_ground.front().z(),.7,1e-12); EXPECT_GT(result.ground.cell_z.size(),0U);}

TEST(Geometry, GroundRemovalRejectsInvalidLocalGroundParameters) {
  Points p = plane_points();
  GroundParameters params;
  params.local_cell_size = 0.0;
  EXPECT_THROW(remove_ground(p, params), std::invalid_argument);
  params.local_cell_size = .5;
  params.local_clearance = -.01;
  EXPECT_THROW(remove_ground(p, params), std::invalid_argument);
}

}}  // namespace blockpose::core
