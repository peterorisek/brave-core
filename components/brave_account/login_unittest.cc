/* Copyright (c) 2026 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/. */

#include "brave/components/brave_account/login.h"

#include <optional>
#include <string>

#include "base/functional/bind.h"
#include "base/functional/callback.h"
#include "base/no_destructor.h"
#include "base/test/task_environment.h"
#include "base/types/expected.h"
#include "base/values.h"
#include "brave/components/brave_account/brave_account_service_test.h"
#include "brave/components/brave_account/brave_account_state_prefs.h"
#include "brave/components/brave_account/endpoints/login_finalize.h"
#include "brave/components/brave_account/endpoints/login_init.h"
#include "brave/components/brave_account/mojom/brave_account.mojom.h"
#include "components/prefs/pref_service.h"
#include "mojo/public/cpp/bindings/remote.h"
#include "net/http/http_status_code.h"
#include "testing/gtest/include/gtest/gtest.h"

namespace brave_account {

using endpoints::LoginFinalize;
using endpoints::LoginInit;

struct LoginInitTestCase {
  using Endpoint = LoginInit;
  using EndpointResponse = Endpoint::Response;
  using MojoExpected =
      base::expected<mojom::LoginInitResultPtr, mojom::LoginErrorPtr>;

  static void Run(const LoginInitTestCase& test_case,
                  PrefService& pref_service,
                  base::test::TaskEnvironment& task_environment,
                  mojo::Remote<mojom::Authentication>& authentication,
                  base::OnceCallback<void(MojoExpected)> callback) {
    authentication->LoginInit(mojom::Service::kAccounts, test_case.email,
                              test_case.serialized_ke1, std::move(callback));
  }

  std::string test_name;
  std::string email;
  std::string serialized_ke1;
  bool fail_encryption;
  bool fail_decryption;
  std::optional<EndpointResponse> endpoint_response;
  MojoExpected mojo_expected;
};

namespace {

const LoginInitTestCase* LoginInitBodyMissingOrFailedToParse() {
  static const base::NoDestructor<LoginInitTestCase>
      kLoginInitBodyMissingOrFailedToParse({
          .test_name = "login_init_body_missing_or_failed_to_parse",
          .email = kEmailAddress,
          .serialized_ke1 = "serialized_ke1",
          .fail_encryption = {},  // not used
          .fail_decryption = {},  // not used
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_INTERNAL_SERVER_ERROR,
                                 .body = std::nullopt}},
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_INTERNAL_SERVER_ERROR,
                  mojom::LoginServerErrorCode::kInvalidResponse))),
      });
  return kLoginInitBodyMissingOrFailedToParse.get();
}

const LoginInitTestCase* LoginInitErrorCodeIsNull() {
  static const base::NoDestructor<LoginInitTestCase> kLoginInitErrorCodeIsNull({
      .test_name = "login_init_error_code_is_null",
      .email = kEmailAddress,
      .serialized_ke1 = "serialized_ke1",
      .fail_encryption = {},  // not used
      .fail_decryption = {},  // not used
      .endpoint_response = {{.net_error = net::OK,
                             .status_code = net::HTTP_BAD_REQUEST,
                             .body = base::unexpected([] {
                               LoginInit::Response::ErrorBody body;
                               body.code = base::Value();
                               return body;
                             }())}},
      .mojo_expected = base::unexpected(
          mojom::LoginError::NewServerError(mojom::LoginServerError::New(
              net::HTTP_BAD_REQUEST, mojom::LoginServerErrorCode::kNull))),
  });
  return kLoginInitErrorCodeIsNull.get();
}

const LoginInitTestCase* LoginInitEmailNotVerified() {
  static const base::NoDestructor<LoginInitTestCase> kLoginInitEmailNotVerified(
      {
          .test_name = "login_init_email_not_verified",
          .email = kEmailAddress,
          .serialized_ke1 = "serialized_ke1",
          .fail_encryption = {},  // not used
          .fail_decryption = {},  // not used
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_UNAUTHORIZED,
                                 .body = base::unexpected([] {
                                   LoginInit::Response::ErrorBody body;
                                   body.code = base::Value(11003);
                                   return body;
                                 }())}},
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_UNAUTHORIZED,
                  mojom::LoginServerErrorCode::kEmailNotVerified))),
      });
  return kLoginInitEmailNotVerified.get();
}

const LoginInitTestCase* LoginInitIncorrectCredentials() {
  static const base::NoDestructor<LoginInitTestCase>
      kLoginInitIncorrectCredentials({
          .test_name = "login_init_incorrect_credentials",
          .email = kEmailAddress,
          .serialized_ke1 = "serialized_ke1",
          .fail_encryption = {},  // not used
          .fail_decryption = {},  // not used
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_UNAUTHORIZED,
                                 .body = base::unexpected([] {
                                   LoginInit::Response::ErrorBody body;
                                   body.code = base::Value(14004);
                                   return body;
                                 }())}},
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_UNAUTHORIZED,
                  mojom::LoginServerErrorCode::kIncorrectCredentials))),
      });
  return kLoginInitIncorrectCredentials.get();
}

const LoginInitTestCase* LoginInitIncorrectEmail() {
  static const base::NoDestructor<LoginInitTestCase> kLoginInitIncorrectEmail({
      .test_name = "login_init_incorrect_email",
      .email = kEmailAddress,
      .serialized_ke1 = "serialized_ke1",
      .fail_encryption = {},  // not used
      .fail_decryption = {},  // not used
      .endpoint_response = {{.net_error = net::OK,
                             .status_code = net::HTTP_UNAUTHORIZED,
                             .body = base::unexpected([] {
                               LoginInit::Response::ErrorBody body;
                               body.code = base::Value(14005);
                               return body;
                             }())}},
      .mojo_expected = base::unexpected(
          mojom::LoginError::NewServerError(mojom::LoginServerError::New(
              net::HTTP_UNAUTHORIZED,
              mojom::LoginServerErrorCode::kIncorrectEmail))),
  });
  return kLoginInitIncorrectEmail.get();
}

const LoginInitTestCase* LoginInitIncorrectPassword() {
  static const base::NoDestructor<LoginInitTestCase>
      kLoginInitIncorrectPassword({
          .test_name = "login_init_incorrect_password",
          .email = kEmailAddress,
          .serialized_ke1 = "serialized_ke1",
          .fail_encryption = {},  // not used
          .fail_decryption = {},  // not used
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_UNAUTHORIZED,
                                 .body = base::unexpected([] {
                                   LoginInit::Response::ErrorBody body;
                                   body.code = base::Value(14006);
                                   return body;
                                 }())}},
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_UNAUTHORIZED,
                  mojom::LoginServerErrorCode::kIncorrectPassword))),
      });
  return kLoginInitIncorrectPassword.get();
}

const LoginInitTestCase* LoginInitServerError() {
  static const base::NoDestructor<LoginInitTestCase> kLoginInitServerError({
      .test_name = "login_init_server_error",
      .email = kEmailAddress,
      .serialized_ke1 = "serialized_ke1",
      .fail_encryption = {},  // not used
      .fail_decryption = {},  // not used
      .endpoint_response = {{.net_error = net::OK,
                             .status_code = net::HTTP_INTERNAL_SERVER_ERROR,
                             .body = base::unexpected([] {
                               LoginInit::Response::ErrorBody body;
                               body.code = base::Value(0);
                               return body;
                             }())}},
      .mojo_expected = base::unexpected(mojom::LoginError::NewServerError(
          mojom::LoginServerError::New(net::HTTP_INTERNAL_SERVER_ERROR,
                                       mojom::LoginServerErrorCode::kNull))),
  });
  return kLoginInitServerError.get();
}

const LoginInitTestCase* LoginInitUnknown() {
  static const base::NoDestructor<LoginInitTestCase> kLoginInitUnknown({
      .test_name = "login_init_unknown",
      .email = kEmailAddress,
      .serialized_ke1 = "serialized_ke1",
      .fail_encryption = {},  // not used
      .fail_decryption = {},  // not used
      .endpoint_response = {{.net_error = net::OK,
                             .status_code = net::HTTP_TOO_EARLY,
                             .body = base::unexpected([] {
                               LoginInit::Response::ErrorBody body;
                               body.code = base::Value(42);
                               return body;
                             }())}},
      .mojo_expected = base::unexpected(
          mojom::LoginError::NewServerError(mojom::LoginServerError::New(
              net::HTTP_TOO_EARLY, mojom::LoginServerErrorCode::kUnknown))),
  });
  return kLoginInitUnknown.get();
}

const LoginInitTestCase* LoginInitLoginTokenEmpty() {
  static const base::NoDestructor<LoginInitTestCase> kLoginInitLoginTokenEmpty({
      .test_name = "login_init_login_token_empty",
      .email = kEmailAddress,
      .serialized_ke1 = "serialized_ke1",
      .fail_encryption = {},  // not used
      .fail_decryption = {},  // not used
      .endpoint_response = {{.net_error = net::OK,
                             .status_code = net::HTTP_OK,
                             .body =
                                 [] {
                                   LoginInit::Response::SuccessBody body;
                                   body.login_token = "";
                                   body.serialized_ke2 = "serialized_ke2";
                                   return body;
                                 }()}},
      .mojo_expected = base::unexpected(
          mojom::LoginError::NewServerError(mojom::LoginServerError::New(
              net::HTTP_OK, mojom::LoginServerErrorCode::kInvalidResponse))),
  });
  return kLoginInitLoginTokenEmpty.get();
}

const LoginInitTestCase* LoginInitSerializedKe2Empty() {
  static const base::NoDestructor<LoginInitTestCase>
      kLoginInitSerializedKe2Empty({
          .test_name = "login_init_serialized_ke2_empty",
          .email = kEmailAddress,
          .serialized_ke1 = "serialized_ke1",
          .fail_encryption = {},  // not used
          .fail_decryption = {},  // not used
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_OK,
                                 .body =
                                     [] {
                                       LoginInit::Response::SuccessBody body;
                                       body.login_token = kLoginToken;
                                       body.serialized_ke2 = "";
                                       return body;
                                     }()}},
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_OK,
                  mojom::LoginServerErrorCode::kInvalidResponse))),
      });
  return kLoginInitSerializedKe2Empty.get();
}

const LoginInitTestCase* LoginInitLoginTokenFailedToEncrypt() {
  static const base::NoDestructor<LoginInitTestCase>
      kLoginInitLoginTokenFailedToEncrypt({
          .test_name = "login_init_login_token_failed_to_encrypt",
          .email = kEmailAddress,
          .serialized_ke1 = "serialized_ke1",
          .fail_encryption = true,
          .fail_decryption = {},  // not used
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_OK,
                                 .body =
                                     [] {
                                       LoginInit::Response::SuccessBody body;
                                       body.login_token = kLoginToken;
                                       body.serialized_ke2 = "serialized_ke2";
                                       return body;
                                     }()}},
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewClientError(mojom::LoginClientError::New(
                  mojom::LoginClientErrorCode::kLoginTokenEncryptionFailed))),
      });
  return kLoginInitLoginTokenFailedToEncrypt.get();
}

const LoginInitTestCase* LoginInitSuccess() {
  static const base::NoDestructor<LoginInitTestCase> kLoginInitSuccess({
      .test_name = "login_init_success",
      .email = kEmailAddress,
      .serialized_ke1 = "serialized_ke1",
      .fail_encryption = false,
      .fail_decryption = {},  // not used
      .endpoint_response = {{.net_error = net::OK,
                             .status_code = net::HTTP_OK,
                             .body =
                                 [] {
                                   LoginInit::Response::SuccessBody body;
                                   body.login_token = kLoginToken;
                                   body.serialized_ke2 = "serialized_ke2";
                                   return body;
                                 }()}},
      .mojo_expected =
          mojom::LoginInitResult::New(EncryptedLoginToken(), "serialized_ke2"),
  });
  return kLoginInitSuccess.get();
}

using BraveAccountServiceLoginInitTest =
    BraveAccountServiceTest<LoginInitTestCase>;

}  // namespace

TEST_P(BraveAccountServiceLoginInitTest, MapsEndpointExpectedToMojoExpected) {
  RunTestCase();
}

INSTANTIATE_TEST_SUITE_P(BraveAccountServiceTests,
                         BraveAccountServiceLoginInitTest,
                         testing::Values(LoginInitBodyMissingOrFailedToParse(),
                                         LoginInitErrorCodeIsNull(),
                                         LoginInitEmailNotVerified(),
                                         LoginInitIncorrectCredentials(),
                                         LoginInitIncorrectEmail(),
                                         LoginInitIncorrectPassword(),
                                         LoginInitServerError(),
                                         LoginInitUnknown(),
                                         LoginInitLoginTokenEmpty(),
                                         LoginInitSerializedKe2Empty(),
                                         LoginInitLoginTokenFailedToEncrypt(),
                                         LoginInitSuccess()),
                         BraveAccountServiceLoginInitTest::kNameGenerator);

struct LoginFinalizeTestCase {
  using Endpoint = LoginFinalize;
  using EndpointResponse = Endpoint::Response;
  using MojoExpected =
      base::expected<mojom::LoginFinalizeResultPtr, mojom::LoginErrorPtr>;

  static void Run(const LoginFinalizeTestCase& test_case,
                  PrefService& pref_service,
                  base::test::TaskEnvironment& task_environment,
                  mojo::Remote<mojom::Authentication>& authentication,
                  base::OnceCallback<void(MojoExpected)> callback) {
    authentication->LoginFinalize(
        test_case.encrypted_login_token, test_case.client_mac,
        std::move(callback).Then(base::BindOnce(
            [](PrefService* pref_service, std::string expected_email,
               std::string expected_authentication_token) {
              AccountStatePrefs account_state_prefs(*pref_service);
              const auto state = account_state_prefs.GetAccountState();
              if (expected_authentication_token.empty()) {
                EXPECT_TRUE(state->is_logged_out());
              } else {
                ASSERT_TRUE(state->is_logged_in());
                EXPECT_EQ(state->get_logged_in()->email, expected_email);
                EXPECT_EQ(account_state_prefs.GetAuthenticationToken(),
                          expected_authentication_token);
              }
            },
            base::Unretained(&pref_service), test_case.expected_email,
            test_case.expected_authentication_token)));
  }

  std::string test_name;
  std::string encrypted_login_token;
  std::string client_mac;
  bool fail_encryption;
  bool fail_decryption;
  std::optional<EndpointResponse> endpoint_response;
  std::string expected_email;
  std::string expected_authentication_token;
  MojoExpected mojo_expected;
};

namespace {

const LoginFinalizeTestCase* LoginFinalizeLoginTokenFailedToDecrypt() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeLoginTokenFailedToDecrypt({
          .test_name = "login_finalize_login_token_failed_to_decrypt",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = true,
          .endpoint_response = {},  // not used
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewClientError(mojom::LoginClientError::New(
                  mojom::LoginClientErrorCode::kLoginTokenDecryptionFailed))),
      });
  return kLoginFinalizeLoginTokenFailedToDecrypt.get();
}

const LoginFinalizeTestCase* LoginFinalizeBodyMissingOrFailedToParse() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeBodyMissingOrFailedToParse({
          .test_name = "login_finalize_body_missing_or_failed_to_parse",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_INTERNAL_SERVER_ERROR,
                                 .body = std::nullopt}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_INTERNAL_SERVER_ERROR,
                  mojom::LoginServerErrorCode::kInvalidResponse))),
      });
  return kLoginFinalizeBodyMissingOrFailedToParse.get();
}

const LoginFinalizeTestCase* LoginFinalizeErrorCodeIsNull() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeErrorCodeIsNull({
          .test_name = "login_finalize_error_code_is_null",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_BAD_REQUEST,
                                 .body = base::unexpected([] {
                                   LoginFinalize::Response::ErrorBody body;
                                   body.code = base::Value();
                                   return body;
                                 }())}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_BAD_REQUEST, mojom::LoginServerErrorCode::kNull))),
      });
  return kLoginFinalizeErrorCodeIsNull.get();
}

const LoginFinalizeTestCase* LoginFinalizeInterimPasswordStateMismatch() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeInterimPasswordStateMismatch({
          .test_name = "login_finalize_interim_password_state_mismatch",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_BAD_REQUEST,
                                 .body = base::unexpected([] {
                                   LoginFinalize::Response::ErrorBody body;
                                   body.code = base::Value(14009);
                                   return body;
                                 }())}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_BAD_REQUEST,
                  mojom::LoginServerErrorCode::kInterimPasswordStateMismatch))),
      });
  return kLoginFinalizeInterimPasswordStateMismatch.get();
}

const LoginFinalizeTestCase* LoginFinalizeInterimPasswordStateNotFound() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeInterimPasswordStateNotFound({
          .test_name = "login_finalize_interim_password_state_not_found",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_UNAUTHORIZED,
                                 .body = base::unexpected([] {
                                   LoginFinalize::Response::ErrorBody body;
                                   body.code = base::Value(14001);
                                   return body;
                                 }())}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_UNAUTHORIZED,
                  mojom::LoginServerErrorCode::kInterimPasswordStateNotFound))),
      });
  return kLoginFinalizeInterimPasswordStateNotFound.get();
}

const LoginFinalizeTestCase* LoginFinalizeInterimPasswordStateHasExpired() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeInterimPasswordStateHasExpired({
          .test_name = "login_finalize_interim_password_state_has_expired",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_UNAUTHORIZED,
                                 .body = base::unexpected([] {
                                   LoginFinalize::Response::ErrorBody body;
                                   body.code = base::Value(14002);
                                   return body;
                                 }())}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_UNAUTHORIZED,
                  mojom::LoginServerErrorCode::
                      kInterimPasswordStateHasExpired))),
      });
  return kLoginFinalizeInterimPasswordStateHasExpired.get();
}

const LoginFinalizeTestCase* LoginFinalizeIncorrectCredentials() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeIncorrectCredentials({
          .test_name = "login_finalize_incorrect_credentials",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_UNAUTHORIZED,
                                 .body = base::unexpected([] {
                                   LoginFinalize::Response::ErrorBody body;
                                   body.code = base::Value(14004);
                                   return body;
                                 }())}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_UNAUTHORIZED,
                  mojom::LoginServerErrorCode::kIncorrectCredentials))),
      });
  return kLoginFinalizeIncorrectCredentials.get();
}

const LoginFinalizeTestCase* LoginFinalizeIncorrectEmail() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeIncorrectEmail({
          .test_name = "login_finalize_incorrect_email",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_UNAUTHORIZED,
                                 .body = base::unexpected([] {
                                   LoginFinalize::Response::ErrorBody body;
                                   body.code = base::Value(14005);
                                   return body;
                                 }())}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_UNAUTHORIZED,
                  mojom::LoginServerErrorCode::kIncorrectEmail))),
      });
  return kLoginFinalizeIncorrectEmail.get();
}

const LoginFinalizeTestCase* LoginFinalizeIncorrectPassword() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeIncorrectPassword({
          .test_name = "login_finalize_incorrect_password",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_UNAUTHORIZED,
                                 .body = base::unexpected([] {
                                   LoginFinalize::Response::ErrorBody body;
                                   body.code = base::Value(14006);
                                   return body;
                                 }())}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_UNAUTHORIZED,
                  mojom::LoginServerErrorCode::kIncorrectPassword))),
      });
  return kLoginFinalizeIncorrectPassword.get();
}

const LoginFinalizeTestCase* LoginFinalizeServerError() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeServerError({
          .test_name = "login_finalize_server_error",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_INTERNAL_SERVER_ERROR,
                                 .body = base::unexpected([] {
                                   LoginFinalize::Response::ErrorBody body;
                                   body.code = base::Value(0);
                                   return body;
                                 }())}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_INTERNAL_SERVER_ERROR,
                  mojom::LoginServerErrorCode::kNull))),
      });
  return kLoginFinalizeServerError.get();
}

const LoginFinalizeTestCase* LoginFinalizeUnknown() {
  static const base::NoDestructor<LoginFinalizeTestCase> kLoginFinalizeUnknown({
      .test_name = "login_finalize_unknown",
      .encrypted_login_token = EncryptedLoginToken(),
      .client_mac = "client_mac",
      .fail_encryption = {},  // not used
      .fail_decryption = false,
      .endpoint_response = {{.net_error = net::OK,
                             .status_code = net::HTTP_TOO_EARLY,
                             .body = base::unexpected([] {
                               LoginFinalize::Response::ErrorBody body;
                               body.code = base::Value(42);
                               return body;
                             }())}},
      .expected_email = "",
      .expected_authentication_token = "",
      .mojo_expected = base::unexpected(
          mojom::LoginError::NewServerError(mojom::LoginServerError::New(
              net::HTTP_TOO_EARLY, mojom::LoginServerErrorCode::kUnknown))),
  });
  return kLoginFinalizeUnknown.get();
}

const LoginFinalizeTestCase* LoginFinalizeAuthTokenEmpty() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeAuthTokenEmpty({
          .test_name = "login_finalize_auth_token_empty",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_OK,
                                 .body =
                                     [] {
                                       LoginFinalize::Response::SuccessBody
                                           body;
                                       body.auth_token = "";
                                       body.email = kEmailAddress;
                                       return body;
                                     }()}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_OK,
                  mojom::LoginServerErrorCode::kInvalidResponse))),
      });
  return kLoginFinalizeAuthTokenEmpty.get();
}

const LoginFinalizeTestCase* LoginFinalizeEmailEmpty() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeEmailEmpty({
          .test_name = "login_finalize_email_empty",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = {},  // not used
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_OK,
                                 .body =
                                     [] {
                                       LoginFinalize::Response::SuccessBody
                                           body;
                                       body.auth_token = "auth_token";
                                       body.email = "";
                                       return body;
                                     }()}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewServerError(mojom::LoginServerError::New(
                  net::HTTP_OK,
                  mojom::LoginServerErrorCode::kInvalidResponse))),
      });
  return kLoginFinalizeEmailEmpty.get();
}

const LoginFinalizeTestCase* LoginFinalizeAuthenticationTokenFailedToEncrypt() {
  static const base::NoDestructor<LoginFinalizeTestCase>
      kLoginFinalizeAuthenticationTokenFailedToEncrypt({
          .test_name = "login_finalize_authentication_token_failed_to_encrypt",
          .encrypted_login_token = EncryptedLoginToken(),
          .client_mac = "client_mac",
          .fail_encryption = true,
          .fail_decryption = false,
          .endpoint_response = {{.net_error = net::OK,
                                 .status_code = net::HTTP_OK,
                                 .body =
                                     [] {
                                       LoginFinalize::Response::SuccessBody
                                           body;
                                       body.auth_token = "auth_token";
                                       body.email = kEmailAddress;
                                       return body;
                                     }()}},
          .expected_email = "",
          .expected_authentication_token = "",
          .mojo_expected = base::unexpected(
              mojom::LoginError::NewClientError(mojom::LoginClientError::New(
                  mojom::LoginClientErrorCode::
                      kAuthenticationTokenEncryptionFailed))),
      });
  return kLoginFinalizeAuthenticationTokenFailedToEncrypt.get();
}

const LoginFinalizeTestCase* LoginFinalizeSuccess() {
  static const base::NoDestructor<LoginFinalizeTestCase> kLoginFinalizeSuccess({
      .test_name = "login_finalize_success",
      .encrypted_login_token = EncryptedLoginToken(),
      .client_mac = "client_mac",
      .fail_encryption = false,
      .fail_decryption = false,
      .endpoint_response = {{.net_error = net::OK,
                             .status_code = net::HTTP_OK,
                             .body =
                                 [] {
                                   LoginFinalize::Response::SuccessBody body;
                                   body.auth_token = kAuthenticationToken;
                                   body.email = kEmailAddress;
                                   return body;
                                 }()}},
      .expected_email = kEmailAddress,
      .expected_authentication_token = EncryptedAuthenticationToken(),
      .mojo_expected = mojom::LoginFinalizeResult::New(),
  });
  return kLoginFinalizeSuccess.get();
}

using BraveAccountServiceLoginFinalizeTest =
    BraveAccountServiceTest<LoginFinalizeTestCase>;

}  // namespace

TEST_P(BraveAccountServiceLoginFinalizeTest,
       MapsEndpointExpectedToMojoExpected) {
  RunTestCase();
}

INSTANTIATE_TEST_SUITE_P(
    BraveAccountServiceTests,
    BraveAccountServiceLoginFinalizeTest,
    testing::Values(LoginFinalizeLoginTokenFailedToDecrypt(),
                    LoginFinalizeBodyMissingOrFailedToParse(),
                    LoginFinalizeErrorCodeIsNull(),
                    LoginFinalizeInterimPasswordStateMismatch(),
                    LoginFinalizeInterimPasswordStateNotFound(),
                    LoginFinalizeInterimPasswordStateHasExpired(),
                    LoginFinalizeIncorrectCredentials(),
                    LoginFinalizeIncorrectEmail(),
                    LoginFinalizeIncorrectPassword(),
                    LoginFinalizeServerError(),
                    LoginFinalizeUnknown(),
                    LoginFinalizeAuthTokenEmpty(),
                    LoginFinalizeEmailEmpty(),
                    LoginFinalizeAuthenticationTokenFailedToEncrypt(),
                    LoginFinalizeSuccess()),
    BraveAccountServiceLoginFinalizeTest::kNameGenerator);

}  // namespace brave_account
