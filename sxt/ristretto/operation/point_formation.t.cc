#include "sxt/ristretto/operation/point_formation.h"

#include <iostream>

#include "sxt/base/test/unit_test.h"

#include "sxt/curve21/property/curve.h"
#include "sxt/curve21/type/element_p3.h"

#include "sxt/field51/type/element.h"

using namespace sxt;
using namespace sxt::rsto;

TEST_CASE("we can form a ristretto point from two field elements") {
  c21t::element_p3 p;

  SECTION("verify the formed point is on the curve") {
    f51t::element r0{1, 0, 0, 0, 0};
    f51t::element r1{0, 1, 0, 0, 0};
    form_ristretto_point(p, r0, r1);
    REQUIRE(c21p::is_on_curve(p));
  }

  SECTION("verify against values from libsodium") {
    f51t::element r0{884718643633428, 762319459268479, 1709312581761443, 1279368164090145,
                     1358805166752477};
    f51t::element r1{655729438436755, 1749471821978280, 1423375735888210, 2232173112283343,
                     625898374778055};
    form_ristretto_point(p, r0, r1);
    REQUIRE(c21p::is_on_curve(p));
    c21t::element_p3 expected_p{
        .X = {1414777681828309, 656687082230489, 1050027234858999, 1537949840354010,
              1851452595745482},
        .Y = {330216608683317, 1882825755417785, 485917604015296, 1752809424055599,
              1382139156449365},
        .Z = {857732158257297, 2042606526776349, 1877199191228704, 2019396108674854,
              54862903668982},
        .T = {344303365790600, 621298452801063, 946706483578814, 276019194381015, 1092985771077319},
    };
    REQUIRE(p == expected_p);
  }
}
